from pathlib import Path
from PIL import Image
import numpy as np
import json
import csv
import re

# -------------------------
# 사용자 환경 설정
# -------------------------
DATASET_ROOT = Path("dataset")
RAW_ROOT = DATASET_ROOT / "5ms"
IMAGES_ROOT = Path("dataset/images") 

# 입력 원본 폴더(단일)
RAW_DIR = RAW_ROOT / "raw"          # frame_000.png ...
DENOISED_DIR = RAW_ROOT / "denoised"  # out_000.png ...

# 좌표/라벨(글로벌)
POSITIONS_PATH = RAW_ROOT / "position9.npy"  # (n_sites,2) (y,x)
LABELS_PATH = RAW_ROOT / "label9.npy"        # 또는 .json

# 크롭 16×16
CROP_SIZE = 16
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
SPLITS = ("train", "val", "test")
RNG_SEED = 42  # 프레임 랜덤 분할 시드

# -------------------------
# 파일명에서 프레임 인덱스 추출
# -------------------------
def parse_raw_index(p: Path) -> int | None:
    m = re.match(r"frame_(\d+)\.png$", p.name)
    return int(m.group(1)) if m else None

def parse_denoised_index(p: Path) -> int | None:
    m = re.match(r"out_(\d+)\.png$", p.name)
    return int(m.group(1)) if m else None

# -------------------------
# 라벨/포지션 로더
# -------------------------
def load_labels(path: Path) -> np.ndarray:
    """(n_frames, n_sites) -> bool"""
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        truth_map = data.get("truth", {})
        items = sorted(truth_map.items(), key=lambda kv: int(kv[0].split("_")[-1]))
        truth_list = [np.asarray(entry.get("truth")) for _, entry in items]
        truth = np.stack(truth_list, axis=0)
        return truth.astype(bool)
    elif path.suffix.lower() == ".npy":
        truth = np.load(str(path))
        if truth.ndim != 2:
            raise ValueError(f"Loaded truth array must be 2D. Got shape {truth.shape}")
        return truth.astype(bool)
    else:
        raise ValueError("Unsupported labels file format. Use .npy or .json")

def load_positions(path: Path) -> np.ndarray:
    """(n_sites,2) int (y,x)"""
    if not path.exists():
        raise FileNotFoundError(f"Positions file not found: {path}")
    if path.suffix.lower() == ".npy":
        pos = np.load(str(path))
    elif path.suffix.lower() == ".csv":
        try:
            pos = np.loadtxt(str(path), delimiter=",", dtype=float)
        except Exception:
            pos = np.loadtxt(str(path), dtype=float)
    else:
        raise ValueError("Unsupported positions file format. Use .npy or .csv")
    pos = np.asarray(pos)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"Positions must have shape (n,2). Got {pos.shape}")
    return np.round(pos).astype(int)

# -------------------------
# edge 복제 패딩 중심 크롭 (16×16)
# -------------------------
def crop_center_16(img: Image.Image, y: int, x: int) -> Image.Image:
    half = CROP_SIZE // 2  # 8
    left, upper = x - half, y - half
    right, lower = x + half, y + half

    if left >= 0 and upper >= 0 and right <= img.width and lower <= img.height:
        return img.crop((left, upper, right, lower))

    out = Image.new(img.mode, (CROP_SIZE, CROP_SIZE))
    for dy in range(CROP_SIZE):
        for dx in range(CROP_SIZE):
            src_x = min(max(left + dx, 0), img.width - 1)
            src_y = min(max(upper + dy, 0), img.height - 1)
            out.putpixel((dx, dy), img.getpixel((src_x, src_y)))
    return out

# -------------------------
# 메인 파이프라인 (단일 폴더 → split 생성)
# -------------------------
def main():
    # 인덱스 매핑
    raw_map = {}
    for p in sorted(RAW_DIR.glob("*.png")):
        idx = parse_raw_index(p)
        if idx is not None:
            raw_map[idx] = p

    den_map = {}
    for p in sorted(DENOISED_DIR.glob("*.png")):
        idx = parse_denoised_index(p)
        if idx is not None:
            den_map[idx] = p

    if not raw_map or not den_map:
        raise RuntimeError("raw/denoised 폴더에서 매칭할 PNG가 없습니다.")

    common_frames = sorted(set(raw_map.keys()) & set(den_map.keys()))
    if not common_frames:
        raise RuntimeError("raw와 denoised 사이에 공통 프레임 인덱스가 없습니다.")

    # 라벨/포지션 로딩
    positions = load_positions(POSITIONS_PATH)   # (n_sites,2)
    truth = load_labels(LABELS_PATH)             # (n_frames,n_sites) bool
    n_sites = positions.shape[0]
    n_frames = truth.shape[0]

    max_idx = max(common_frames)
    if max_idx >= n_frames:
        print(f"[경고] truth 프레임 수({n_frames}) < 최대 파일 인덱스({max_idx}). 겹치는 구간만 처리합니다.")

    # 프레임 단위 분할
    frames = np.array(common_frames, dtype=int)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(frames)

    n = len(frames)
    n_train = int(round(n * SPLIT_RATIOS[0]))
    n_val = int(round(n * SPLIT_RATIOS[1]))
    # 남은 건 test
    n_test = n - n_train - n_val

    train_frames = set(perm[:n_train].tolist())
    val_frames   = set(perm[n_train:n_train+n_val].tolist())
    test_frames  = set(perm[n_train+n_val:].tolist())

    split_of = {}
    for f in train_frames: split_of[f] = "train"
    for f in val_frames:   split_of[f] = "val"
    for f in test_frames:  split_of[f] = "test"

    # split별 출력 폴더 & CSV 준비
    out_dirs = {}
    writers = {}
    csv_files = {}
    for split in SPLITS:
        base = IMAGES_ROOT / split
        raw_out_dir = base / "raw_crops_16"
        den_out_dir = base / "denoised_crops_16"
        raw_out_dir.mkdir(parents=True, exist_ok=True)
        den_out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[split] = (raw_out_dir, den_out_dir)

        csv_path = base / "crops_16_pairs.csv"
        f = open(csv_path, "w", newline="", encoding="utf-8")
        csv_files[split] = f
        w = csv.writer(f)
        w.writerow(["id", "raw_path", "denoised_path", "label"])
        writers[split] = w

    # 처리
    for frame_idx in common_frames:
        if frame_idx >= n_frames:
            continue
        split = split_of[frame_idx]
        raw_img = Image.open(raw_map[frame_idx])
        den_img = Image.open(den_map[frame_idx])

        raw_out_dir, den_out_dir = out_dirs[split]
        writer = writers[split]

        for site_idx, (y, x) in enumerate(positions):
            raw_crop = crop_center_16(raw_img, int(y), int(x))
            den_crop = crop_center_16(den_img, int(y), int(x))
            label = int(bool(truth[frame_idx, site_idx]))  # 0/1

            uid = f"{frame_idx:06d}_{site_idx:03d}"
            raw_out_path = raw_out_dir / f"{uid}.png"
            den_out_path = den_out_dir / f"{uid}.png"
            raw_crop.save(raw_out_path)
            den_crop.save(den_out_path)

            # CSV에는 dataset/ 기준 상대경로 기록
            raw_rel = raw_out_path.relative_to(DATASET_ROOT.parent)   # "dataset/..."
            den_rel = den_out_path.relative_to(DATASET_ROOT.parent)
            writer.writerow([uid, str(raw_rel), str(den_rel), label])

    # CSV 닫기
    for f in csv_files.values():
        f.close()

    print("[완료] Splits:")
    print(f"  train: {len(train_frames)} frames")
    print(f"  val:   {len(val_frames)} frames")
    print(f"  test:  {len(test_frames)} frames")
    for split in SPLITS:
        print(f"  {split} CSV -> {IMAGES_ROOT / split / 'crops_16_pairs.csv'}")

if __name__ == "__main__":
    main()