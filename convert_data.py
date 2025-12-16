#!/usr/bin/env python
import os
import sys
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
import h5py
import SimpleITK as sitk
from tqdm import tqdm
from skimage.transform import resize

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    dataset_root: str
    output_root: str
    img_size: int = 224
    clip_percentile: float = 0.995
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 1234

# ============================================================
# DATA CONVERSION UTILITIES
# ============================================================

@dataclass
class CaseInfo:
    """Metadata for a case belonging to a specific center"""
    center: str
    image_path: Path
    label_path: Optional[Path]

    @property
    def base_name(self) -> str:
        name = self.image_path.name
        for suffix in reversed(self.image_path.suffixes):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name

    @property
    def identifier(self) -> str:
        return f"{self.center}_{self.base_name}"

def load_volume(image_path: Path, label_path: Optional[Path]) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
    """Load NIfTI volume and return numpy array + spacing"""
    image_itk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    spacing = image_itk.GetSpacing()

    label_array = None
    if label_path is not None and label_path.exists():
        label_itk = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label_itk).astype(np.int16)
    return image_array, label_array, spacing

def intensity_clip(volume: np.ndarray, percentile: float = 0.995) -> np.ndarray:
    """Clip intensity values at percentile to remove outliers"""
    lower = np.percentile(volume, 0.5)
    upper = np.percentile(volume, percentile * 100)
    return np.clip(volume, lower, upper)

def min_max_normalise(volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Min-Max normalization to [0, 1]"""
    min_val = volume.min()
    max_val = volume.max()
    return (volume - min_val) / (max_val - min_val + epsilon)

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def write_slice_npz(output_path: Path, image_slice: np.ndarray, label_slice: Optional[np.ndarray], target_size=(224, 224)) -> None:
    """Save a 2D slice as .npz, resized to target_size"""
    ensure_dir(output_path.parent)
    
    if image_slice.shape != target_size:
        image_slice = resize(image_slice, target_size, order=1, mode='constant', preserve_range=True)
        if label_slice is not None:
            label_slice = resize(label_slice, target_size, order=0, mode='constant', preserve_range=True)

    save_dict = {'image': image_slice.astype(np.float32)}
    if label_slice is not None:
        save_dict['label'] = label_slice.astype(np.uint8)
        
    np.savez_compressed(output_path, **save_dict)

def write_volume_h5(output_path: Path, image_volume: np.ndarray, label_volume: Optional[np.ndarray], 
                    spacing: Tuple[float, float, float]) -> None:
    """Save full 3D volume as HDF5 (for Val/Test)"""
    ensure_dir(output_path.parent)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("image", data=image_volume.astype(np.float32), compression="gzip")
        if label_volume is not None:
            f.create_dataset("label", data=label_volume.astype(np.int16), compression="gzip")
        f.create_dataset("voxel_spacing", data=np.asarray(spacing, dtype=np.float32), compression="gzip")

def gather_cases_by_center(dataset_root: Path) -> Dict[str, List[CaseInfo]]:
    """Scan directory and gather cases by center"""
    centers: Dict[str, List[CaseInfo]] = {}
    
    if not dataset_root.exists():
         raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for center_dir in sorted(dataset_root.iterdir()):
        if not center_dir.is_dir():
            continue
        image_dir = center_dir / "images"
        label_dir = center_dir / "labels"
        if not image_dir.exists():
            continue
        cases: List[CaseInfo] = []
        for image_path in sorted(image_dir.glob("*.nii*")):
            label_path = label_dir / image_path.name if label_dir.exists() else None
            if label_path is not None and not label_path.exists():
                print(f"⚠️ Missing label for {image_path}")
                continue
            cases.append(CaseInfo(center=center_dir.name, image_path=image_path, label_path=label_path))
        if cases:
            centers[center_dir.name] = cases
    if not centers:
        raise FileNotFoundError(f"No centers found in {dataset_root}")
    return centers

def split_cases_by_ratio(centers: Dict[str, List[CaseInfo]], ratios: Sequence[float], seed: int) -> Tuple[List[CaseInfo], List[CaseInfo], List[CaseInfo]]:
    """Split cases into train/val/test by ratio"""
    rng = np.random.default_rng(seed)
    train_cases, val_cases, test_cases = [], [], []

    for center, cases in centers.items():
        indices = rng.permutation(len(cases))
        raw = np.array(ratios) * len(cases)
        counts = np.floor(raw).astype(int)
        remainder = len(cases) - counts.sum()
        if remainder > 0:
            order = np.argsort(raw - counts)[::-1]
            for idx in order[:remainder]:
                counts[idx] += 1
        
        train_n, val_n, test_n = counts
        train_cases.extend(cases[i] for i in indices[:train_n])
        val_cases.extend(cases[i] for i in indices[train_n:train_n + val_n])
        test_cases.extend(cases[i] for i in indices[train_n + val_n:train_n + val_n + test_n])

    return train_cases, val_cases, test_cases

def convert_data(config: Config, force_reconvert: bool = False):
    """
    Convert NIfTI data to NPZ (Train) and HDF5 (Val/Test)
    """
    output_root = Path(config.output_root)
    training_dir = output_root / "training_set"
    val_dir = output_root / "val_set"
    test_dir = output_root / "test_set"
    
    if not force_reconvert and training_dir.exists() and len(list(training_dir.glob("*.npz"))) > 0:
        n_train = len(list(training_dir.glob("*.npz")))
        n_val = len(list(val_dir.glob("*.h5"))) if val_dir.exists() else 0
        n_test = len(list(test_dir.glob("*.h5"))) if test_dir.exists() else 0
        print(f"✅ Data already converted:")
        print(f"   Training slices (.npz): {n_train}")
        print(f"   Validation volumes (.h5): {n_val}")
        print(f"   Test volumes (.h5): {n_test}")
        return
    
    print("🔄 Starting data conversion...")
    
    dataset_root = Path(config.dataset_root)
    try:
        centers = gather_cases_by_center(dataset_root)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    print(f"\n📊 Found {len(centers)} centers:")
    for center, cases in centers.items():
        print(f"   {center}: {len(cases)} cases")
    
    ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
    train_cases, val_cases, test_cases = split_cases_by_ratio(centers, ratios, config.seed)
    
    print(f"\n📊 Split: {len(train_cases)} train, {len(val_cases)} val, {len(test_cases)} test")
    
    splits_dir = output_root / "splits"
    ensure_dir(splits_dir)
    for split_name, cases in [("train", train_cases), ("val", val_cases), ("test", test_cases)]:
        with open(splits_dir / f"{split_name}.txt", "w") as f:
            for case in cases:
                f.write(case.identifier + "\n")
    
    # --- Convert TRAINING cases ---
    print("\n🔄 Converting training cases to 2D slices (.npz)...")
    ensure_dir(training_dir)
    total_slices = 0
    
    target_size = (config.img_size, config.img_size)
    
    for case in tqdm(train_cases, desc="Training cases"):
        image_volume, label_volume, _ = load_volume(case.image_path, case.label_path)
        image_volume = intensity_clip(image_volume, config.clip_percentile)
        image_volume = min_max_normalise(image_volume)
        
        for idx in range(image_volume.shape[0]):
            slice_id = f"{case.identifier}_slice_{idx:03d}.npz"
            write_slice_npz(
                training_dir / slice_id, 
                image_volume[idx], 
                label_volume[idx],
                target_size=target_size 
            )
            total_slices += 1
    
    print(f"   ✅ Saved {total_slices} training slices (.npz)")
    
    # --- Convert VALIDATION cases ---
    print("\n🔄 Converting validation cases to 3D volumes (.h5)...")
    ensure_dir(val_dir)
    for case in tqdm(val_cases, desc="Validation cases"):
        image_volume, label_volume, spacing = load_volume(case.image_path, case.label_path)
        image_volume = min_max_normalise(intensity_clip(image_volume, config.clip_percentile))
        write_volume_h5(val_dir / f"{case.identifier}.h5", image_volume, label_volume, spacing)
    print(f"   ✅ Saved {len(val_cases)} validation volumes")
    
    # --- Convert TEST cases ---
    if test_cases:
        print("\n🔄 Converting test cases to 3D volumes (.h5)...")
        ensure_dir(test_dir)
        for case in tqdm(test_cases, desc="Test cases"):
            image_volume, label_volume, spacing = load_volume(case.image_path, case.label_path)
            image_volume = min_max_normalise(intensity_clip(image_volume, config.clip_percentile))
            write_volume_h5(test_dir / f"{case.identifier}.h5", image_volume, label_volume, spacing)
        print(f"   ✅ Saved {len(test_cases)} test volumes")
    
    print("\n✅ Data conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI data to NPZ/H5 for Swin-UNet training")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the root directory containing center folders")
    parser.add_argument("--output-root", type=str, default="./outputs/swin_dataset_npz", help="Path to output directory")
    parser.add_argument("--img-size", type=int, default=224, help="Target image size (default: 224)")
    parser.add_argument("--force", action="store_true", help="Force reconversion even if data exists")
    
    args = parser.parse_args()
    
    config = Config(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        img_size=args.img_size
    )
    
    convert_data(config, force_reconvert=args.force)
