# train_atom_cbam.py
import os
import csv
import math
import random
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np

from MODELS.model_atomnet import AtomCBAMNet


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def imread_grayscale(path: str) -> np.ndarray:
    # PIL: L 모드는 8-bit grayscale
    img = Image.open(path).convert("L")
    return np.array(img)  # H×W (uint8 or float)


def to_tensor_2ch_naive(raw_img: np.ndarray, den_img: np.ndarray, size: Tuple[int, int] = (16, 16)) -> torch.Tensor:
    # 리사이즈
    H, W = size
    raw_pil = Image.fromarray(raw_img).resize((W, H), resample=Image.BILINEAR)
    den_pil = Image.fromarray(den_img).resize((W, H), resample=Image.BILINEAR)

    raw = np.asarray(raw_pil).astype(np.float32)
    den = np.asarray(den_pil).astype(np.float32)

    # [0,1] 스케일링
    if raw.max() > 1.0:
        raw /= 255.0
    if den.max() > 1.0:
        den /= 255.0

    raw = np.clip(raw, 0.0, 1.0)
    den = np.clip(den, 0.0, 1.0)

    # 2채널로 쌓기 → (2, H, W)
    stacked = np.stack([raw, den], axis=0)
    return torch.from_numpy(stacked)  # float32 tensor


def to_tensor_2ch(raw_img: np.ndarray,
                  den_img: np.ndarray,
                  size: Tuple[int,int]=(16,16),
                  ring_border: int = 3,
                  do_ring_norm: bool = True) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    import torch

    H, W = size
    # 1) 리사이즈
    raw = np.asarray(Image.fromarray(raw_img).resize((W, H), Image.BILINEAR)).astype(np.float32)
    den = np.asarray(Image.fromarray(den_img).resize((W, H), Image.BILINEAR)).astype(np.float32)

    # 2) [0,1] 스케일 (카메라 uint8/uint16 대응)
    if raw.max() > 1.0: raw /= 255.0
    if den.max() > 1.0: den /= 255.0
    raw = np.clip(raw, 0.0, 1.0)
    den = np.clip(den, 0.0, 1.0)

    # 3) ring 기반 z-score 정규화 (raw에만; den은 그대로 두는 걸 기본값으로)
    if do_ring_norm:
        h, w = raw.shape
        b = max(1, min(ring_border, h // 2, w // 2))
        mask = np.zeros((h, w), dtype=bool)
        mask[:b, :]  = True
        mask[-b:, :] = True
        mask[:, :b]  = True
        mask[:, -b:] = True
        ring = raw[mask]
        mu = float(ring.mean())
        sd = float(ring.std() + 1e-8)
        raw = (raw - mu) / sd
        # ⚠️ 여기서 raw는 z-score이므로 절대 다시 clip(0~1) 하지 마!

    stacked = np.stack([raw, den], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


# 간단한 약한 augmentation (작은 ROI이므로 과하지 않게)
def weak_augment(x: torch.Tensor) -> torch.Tensor:
    # x: (2, H, W)
    # 50% 확률 좌우/상하 flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])  # horizontal
    if random.random() < 0.5:
        x = torch.flip(x, dims=[1])  # vertical
    # 약한 가우시안 노이즈 주입 (아주 작게)
    if random.random() < 0.3:
        noise = torch.randn_like(x) * 0.01
        x = torch.clamp(x + noise, 0.0, 1.0)
    return x


# ---------------------------
# Dataset
# ---------------------------
class AtomTwoChannelDataset(Dataset):
    def __init__(self, csv_path: str, image_size: Tuple[int, int] = (16, 16), augment: bool = False, binary: bool = True):
        super().__init__()
        self.rows = load_csv(csv_path)
        self.image_size = image_size
        self.augment = augment
        self.binary = binary

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw_path = row["raw_path"]
        den_path = row["denoised_path"]
        label = int(row["label"])

        raw = imread_grayscale(raw_path)
        den = imread_grayscale(den_path)
        x = to_tensor_2ch(raw, den, size=self.image_size)  # (2, H, W)

        if self.augment:
            x = weak_augment(x)

        if self.binary:
            y = torch.tensor([float(label)], dtype=torch.float32)  # shape (1,)
        else:
            y = torch.tensor(label, dtype=torch.long)              # scalar class index

        return x, y


# ---------------------------
# Metrics
# ---------------------------
def accuracy_from_logits(logits: torch.Tensor, y, binary: bool) -> float:
    if binary:
        # logits: (B,1), y: (B,1) in {0,1}
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct = (preds == y).sum().item()
        total = y.numel()
        return correct / total
    else:
        # logits: (B,C), y: (B,)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        return correct / total
    
    
# ---------------------------
# EMA Helper (패치 버전)
# ---------------------------
class ModelEMA:
    """
    float 텐서(파라미터/버퍼)만 EMA로 추적.
    - Long/Bool 등 비부동소수 텐서는 스킵 → dtype mismatch 방지
    - 평가 시 apply_shadow()/restore()로 스왑
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}  # name -> tensor(float)
        self.backup = {}  # 원복용

        # state_dict 전체에서 '부동소수'만 추적
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                if tensor.is_floating_point():
                    self.shadow[name] = tensor.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        # 현재 모델의 state_dict를 보며, shadow에 있는 float 항목만 EMA
        msd = model.state_dict()
        for name, tensor in msd.items():
            if name in self.shadow:  # float만 들어있음
                # dtype/device 정합성 유지
                tgt = self.shadow[name]
                self.shadow[name].mul_(self.decay).add_(
                    tensor.detach().to(dtype=tgt.dtype, device=tgt.device) * (1.0 - self.decay)
                )
        # Long/Bool 등은 shadow에 없으므로 자연스레 스킵됨

    def apply_shadow(self, model: nn.Module):
        # shadow의 float 항목만 모델에 덮어씀
        self.backup = {}
        msd = model.state_dict()
        for name, tensor in msd.items():
            if name in self.shadow:
                self.backup[name] = tensor.detach().clone()
                msd[name].copy_(self.shadow[name].to(dtype=msd[name].dtype, device=msd[name].device))

    def restore(self, model: nn.Module):
        # 백업된 항목만 원복
        msd = model.state_dict()
        for name, tensor in self.backup.items():
            msd[name].copy_(tensor)
        self.backup = {}


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    binary: bool = True,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,  # OneCycleLR는 step 단위
    ema: Optional[ModelEMA] = None,
):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # EMA update (step마다)
        if ema is not None:
            ema.update(model)

        # OneCycleLR 등 step 스케줄러라면 여기서 호출
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        batch = x.size(0)
        running_loss += loss.item() * batch
        running_acc += accuracy_from_logits(logits.detach(), y, binary) * batch
        n += batch

    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    binary: bool = True,
    ema: Optional[ModelEMA] = None,  # EMA 가중치로 평가할지 여부
):
    model.eval()

    # EMA로 평가
    using_ema = False
    if ema is not None:
        ema.apply_shadow(model)
        using_ema = True

    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        batch = x.size(0)
        running_loss += loss.item() * batch
        running_acc += accuracy_from_logits(logits, y, binary) * batch
        n += batch

    if using_ema:
        ema.restore(model)

    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate_confusion(model, loader, device, binary=True):
    model.eval()
    TP = TN = FP = FN = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits) if binary else torch.softmax(logits, dim=1)
        preds = (probs >= 0.5).float() if binary else torch.argmax(probs, dim=1)

        if binary:
            # preds, y: shape (B,1) → flatten
            preds = preds.view(-1)
            y = y.view(-1)
            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()
        else:
            raise NotImplementedError("다중 클래스면 말해줘. 그에 맞게 confusion matrix 구성해줄게.")

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    return acc, TP, TN, FP, FN


# ---------------------------
# Main
# ---------------------------
def main(
    train_csv: str = "train.csv",
    val_csv: str = "val.csv",
    test_csv: str = "test.csv",
    num_classes: int = 2,
    image_size: Tuple[int, int] = (16, 16),
    batch_size: int = 128,
    epochs: int = 30,
    base_lr: float = 3e-4,      # AdamW 초기 lr (OneCycle와 함께 사용)
    max_lr: float = 8e-4,       # OneCycleLR max_lr
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    r1: int = 8,
    r2: int = 8,
    use_amp: bool = True,
    seed: int = 1234,
    out_dir: str = "runs/atom_cbam",
    # Early stopping
    es_patience: int = 8,
    es_min_delta: float = 0.002,
    # EMA
    ema_decay: float = 0.999,
):
    """
    (1) Early Stopping + EMA
    (2) OneCycleLR (step 단위 스케줄)
    """
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binary = (num_classes == 1)

    # Datasets / Loaders
    train_ds = AtomTwoChannelDataset(train_csv, image_size=image_size, augment=True, binary=binary)
    val_ds = AtomTwoChannelDataset(val_csv, image_size=image_size, augment=False, binary=binary)
    test_ds = AtomTwoChannelDataset(test_csv, image_size=image_size, augment=False, binary=binary)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = AtomCBAMNet(
        num_classes=num_classes,
        in_channels=2,
        r1=r1, r2=r2,
        dropout=dropout
    ).to(device)

    # Loss
    if binary:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Scheduler: OneCycleLR (step 스케줄)
    steps_per_epoch = max(1, len(train_ld))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        anneal_strategy='cos',
        div_factor=base_lr / max(1e-12, base_lr),  # 기본값 무시: base_lr 그대로 사용
        final_div_factor=1e4,
        three_phase=False
    )

    # AMP 스케일러
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # EMA
    ema = ModelEMA(model, decay=ema_decay)

    # Train loop + Early Stopping
    best_val_acc = -1.0
    best_epoch = 0
    no_improve = 0
    best_path = os.path.join(out_dir, "best.pt")

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_ld, optimizer, loss_fn, device, scaler, binary,
            scheduler=scheduler, ema=ema
        )

        # 검증은 EMA 가중치로 평가 (일반화에 유리)
        va_loss, va_acc = evaluate(model, val_ld, loss_fn, device, binary, ema=ema)

        # Early stopping 로직
        improved = (va_acc - best_val_acc) > es_min_delta
        if improved:
            best_val_acc = va_acc
            best_epoch = ep
            no_improve = 0
            # EMA 가중치로 저장
            ema.apply_shadow(model)
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val_acc": va_acc},
                best_path
            )
            ema.restore(model)
        else:
            no_improve += 1

        # 현재 lr 표시 (OneCycle는 optimizer.param_groups[0]['lr'])
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"[{ep:03d}/{epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {cur_lr:.2e} | "
              f"best@{best_epoch}={best_val_acc:.4f} | "
              f"no_improve={no_improve}")

        if no_improve >= es_patience:
            print("Early stopping triggered.")
            break

    # Load best and test (EMA로 저장된 체크포인트)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} with val_acc={ckpt['val_acc']:.4f}")

    te_loss, te_acc = evaluate(model, test_ld, loss_fn, device, binary)
    acc2, TP, TN, FP, FN = evaluate_confusion(model, test_ld, device, binary=True)

    print(f"[TEST] loss {te_loss:.4f} acc {te_acc:.4f}")
    print(f"Confusion Matrix => TP {TP}, TN {TN}, FP {FP}, FN {FN}, acc_check {acc2:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="./dataset/images/train/crops_16_pairs.csv")
    p.add_argument("--val_csv", type=str, default="./dataset/images/val/crops_16_pairs.csv")
    p.add_argument("--test_csv", type=str, default="./dataset/images/test/crops_16_pairs.csv")
    p.add_argument("--num_classes", type=int, default=2, help="이진이면 1, 다중 클래스는 K")
    p.add_argument("--image_size", type=int, nargs=2, default=(16, 16))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--base_lr", type=float, default=3e-4)
    p.add_argument("--max_lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--r1", type=int, default=8)
    p.add_argument("--r2", type=int, default=8)
    p.add_argument("--no_amp", action="store_true", help="AMP 끄기")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_dir", type=str, default="runs/atom_cbam")
    p.add_argument("--es_patience", type=int, default=8)
    p.add_argument("--es_min_delta", type=float, default=0.002)
    p.add_argument("--ema_decay", type=float, default=0.999)

    args = p.parse_args()

    main(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        num_classes=args.num_classes,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        r1=args.r1,
        r2=args.r2,
        use_amp=not args.no_amp,
        seed=args.seed,
        out_dir=args.out_dir,
        es_patience=args.es_patience,
        es_min_delta=args.es_min_delta,
        ema_decay=args.ema_decay,
    )