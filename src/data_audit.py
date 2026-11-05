"""
Comprehensive Dataset Audit Script for Breast Ultrasound Images (BUSI)
Analyzes file integrity, image-mask pairing, multi-mask cases, dimensions, duplicates,
and class distribution. Generates a reproducible split without data leakage.
"""

import os
import glob
import re
import hashlib
import json
from collections import defaultdict
import numpy as np
from PIL import Image

RAW_DATA_DIR = os.path.join("data", "raw", "Dataset_BUSI_with_GT")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGURES = os.path.join("outputs", "figures")

def compute_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset():
    print("=" * 60)
    print("STARTING AUDIT OF BUSI DATASET")
    print(f"Target Directory: {RAW_DATA_DIR}")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    classes = ["benign", "malignant", "normal"]
    categories = {}
    
    total_images = 0
    total_masks = 0
    corrupt_files = []
    hash_to_files = defaultdict(list)
    cases = []  # list of dicts: {'case_id', 'class', 'image_path', 'mask_paths', 'width', 'height', 'channels'}

    for cls in classes:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        all_pngs = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        
        # In BUSI:
        # Original images: '{class} (idx).png'
        # Masks: '{class} (idx)_mask.png' or '{class} (idx)_mask_1.png'
        img_pattern = re.compile(rf"^{cls}\s+\((\d+)\)\.png$")
        mask_pattern = re.compile(rf"^{cls}\s+\((\d+)\)_mask.*\.png$")

        images_dict = {}
        masks_dict = defaultdict(list)

        for p in all_pngs:
            fname = os.path.basename(p)
            # Check file hash for duplicates
            fhash = compute_hash(p)
            hash_to_files[fhash].append(p)

            # Check if file can be opened
            try:
                with Image.open(p) as img:
                    img.verify()
                # Re-open to read properties (verify leaves file in unreadable state)
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
            except Exception as e:
                corrupt_files.append((p, str(e)))
                continue

            m_match = mask_pattern.match(fname)
            if m_match:
                case_num = int(m_match.group(1))
                masks_dict[case_num].append(p)
                total_masks += 1
            else:
                i_match = img_pattern.match(fname)
                if i_match:
                    case_num = int(i_match.group(1))
                    images_dict[case_num] = {
                        "path": p,
                        "case_id": f"{cls}_{case_num}",
                        "case_num": case_num,
                        "class": cls,
                        "width": w,
                        "height": h,
                        "mode": mode
                    }
                    total_images += 1
                else:
                    print(f"Warning: File did not match expected pattern: {fname}")

        # Pair images with masks
        multi_mask_cases = 0
        missing_mask_cases = 0
        dim_mismatches = []

        for case_num, info in images_dict.items():
            case_masks = sorted(masks_dict.get(case_num, []))
            info["mask_paths"] = case_masks
            info["num_masks"] = len(case_masks)

            if len(case_masks) > 1:
                multi_mask_cases += 1
            elif len(case_masks) == 0:
                missing_mask_cases += 1

            # Check dimension match between image and mask
            for mp in case_masks:
                with Image.open(mp) as m_img:
                    mw, mh = m_img.size
                    if mw != info["width"] or mh != info["height"]:
                        dim_mismatches.append((info["path"], mp, (info["width"], info["height"]), (mw, mh)))

            cases.append(info)

        categories[cls] = {
            "image_count": len(images_dict),
            "mask_count": sum(len(m) for m in masks_dict.values()),
            "multi_mask_cases": multi_mask_cases,
            "missing_mask_cases": missing_mask_cases,
            "dimension_mismatches": len(dim_mismatches)
        }

    # Duplicate check
    duplicates = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}

    # Summary
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Original Images: {total_images}")
    print(f"Total Mask Files: {total_masks}")
    print(f"Corrupted Files: {len(corrupt_files)}")
    print(f"Hash Duplicates: {len(duplicates)}")
    for cls, stats in categories.items():
        print(f"  Class '{cls}':")
        print(f"    Images: {stats['image_count']}")
        print(f"    Masks: {stats['mask_count']}")
        print(f"    Cases with multiple masks: {stats['multi_mask_cases']}")
        print(f"    Cases without masks: {stats['missing_mask_cases']}")
        print(f"    Image-Mask Dimension Mismatches: {stats['dimension_mismatches']}")

    # Check resolution statistics
    widths = [c["width"] for c in cases]
    heights = [c["height"] for c in cases]
    print(f"\nResolution Statistics:")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")

    # Generate Stratified Case-Level Train / Val / Test Split
    # To strictly avoid data leakage, every case_id is atomic.
    # Split: 70% Train, 15% Val, 15% Test
    rng = np.random.default_rng(seed=42)
    train_cases = []
    val_cases = []
    test_cases = []

    for cls in classes:
        cls_cases = [c for c in cases if c["class"] == cls]
        rng.shuffle(cls_cases)
        n = len(cls_cases)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        
        c_train = cls_cases[:n_train]
        c_val = cls_cases[n_train:n_train + n_val]
        c_test = cls_cases[n_train + n_val:]

        for c in c_train:
            c["split"] = "train"
        for c in c_val:
            c["split"] = "val"
        for c in c_test:
            c["split"] = "test"

        train_cases.extend(c_train)
        val_cases.extend(c_val)
        test_cases.extend(c_test)

    print("\n--- STRATIFIED SPLIT DISTRIBUTION ---")
    print(f"Train Cases: {len(train_cases)} ({len(train_cases)/total_images*100:.1f}%)")
    print(f"Val Cases:   {len(val_cases)} ({len(val_cases)/total_images*100:.1f}%)")
    print(f"Test Cases:  {len(test_cases)} ({len(test_cases)/total_images*100:.1f}%)")
    
    for split_name, split_list in [("Train", train_cases), ("Val", val_cases), ("Test", test_cases)]:
        cls_dist = {cls: sum(1 for c in split_list if c["class"] == cls) for cls in classes}
        print(f"  {split_name} class breakdown: {cls_dist}")

    # Save metadata to data/processed
    metadata_path = os.path.join(PROCESSED_DIR, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "provenance": {
                "source": "Breast Ultrasound Images Dataset (BUSI)",
                "authors": "Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, Aly Fahmy",
                "published_in": "Data in Brief, Volume 28, February 2020, 104863",
                "doi": "10.1016/j.dib.2019.104863",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "modality": "B-mode Ultrasound",
                "system": "LOGIQ E9 ultrasound system and LOGIQ 7 system"
            },
            "audit_summary": {
                "total_images": total_images,
                "total_masks": total_masks,
                "classes": categories,
                "corrupt_files_count": len(corrupt_files),
                "duplicate_hashes_count": len(duplicates)
            },
            "cases": cases
        }, f, indent=2)

    print(f"\nMetadata successfully saved to: {metadata_path}")
    return metadata_path

if __name__ == "__main__":
    audit_dataset()
