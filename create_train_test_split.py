import hashlib
import random
from pathlib import Path
import shutil
import sys
import os


SEED = 12345
TEST_FRACTION = 0.2
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def split_user_folder(user_folder: Path, out_base: Path, rng: random.Random):
    # user_folder contains subfolders `face` and `finger`
    user_id = user_folder.name

    for modality in ("face", "finger"):
        src = user_folder / modality
        if not src.exists():
            continue

        images = get_image_files(src)
        if len(images) < 2:
            raise RuntimeError(f"User {user_id} has fewer than 2 images in {modality}; cannot split while keeping user in both sets.")

        rng.shuffle(images)
        test_count = max(1, int(round(len(images) * TEST_FRACTION)))
        test_images = set(images[:test_count])
        train_images = [p for p in images if p not in test_images]

        # destination folders
        train_dest = out_base / "train" / user_id / modality
        test_dest = out_base / "test" / user_id / modality
        train_dest.mkdir(parents=True, exist_ok=True)
        test_dest.mkdir(parents=True, exist_ok=True)

        for p in train_images:
            shutil.copy2(p, train_dest / p.name)

        for p in test_images:
            shutil.copy2(p, test_dest / p.name)


def verify_no_leakage(out_base: Path):
    train_hashes = {}
    test_hashes = {}

    for phase, dest in (("train", out_base / "train"), ("test", out_base / "test")):
        if not dest.exists():
            continue

        for p in dest.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                h = sha256(p)
                (train_hashes if phase == "train" else test_hashes)[h] = p

    intersection = set(train_hashes.keys()).intersection(test_hashes.keys())
    if intersection:
        examples = [(train_hashes[h], test_hashes[h]) for h in list(intersection)[:5]]
        msg = "Train/test leakage detected. Examples:\n"
        for a, b in examples:
            msg += f"  {a} <-> {b}\n"
        raise RuntimeError(msg)


def create_split(dataset_dir: Path, out_base: Path, seed: int = SEED):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset folder not found: {dataset_dir}")

    if out_base.exists():
        raise RuntimeError(f"Output folder already exists: {out_base}. Please remove it before running this script to avoid accidental overwrites.")

    rng = random.Random(seed)

    for user_folder in sorted(dataset_dir.iterdir()):
        if not user_folder.is_dir():
            continue
        # ensure user has face and finger folders
        split_user_folder(user_folder, out_base, rng)

    verify_no_leakage(out_base)


def main():
    repo_root = Path(__file__).parent

    dataset_dir = repo_root / "backend" / "dataset"
    out_base = repo_root / "evaluation_dataset"

    print(f"Reading dataset from: {dataset_dir}")
    print(f"Writing evaluation split to: {out_base}")

    create_split(dataset_dir, out_base)

    print("Done. Created evaluation_dataset with train/test under the repository root.")

if __name__ == "__main__":
    main()
