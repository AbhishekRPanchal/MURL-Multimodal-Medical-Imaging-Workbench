"""
Verify genuine TotalSegmentator CPU inference on a reference non-identifiable CT scan.
Runs real Task 297 (3mm fast model) on external_repos/TotalSegmentator/tests/reference_files/example_ct_sm.nii.gz
Configured for single-thread sequential execution on Windows CPU to prevent multiprocessing spawning issues.
"""

import os
import sys
import time
import numpy as np
import nibabel as nib

def main():
    # Environment setup
    os.environ["TOTALSEG_WEIGHTS_PATH"] = os.path.abspath("checkpoints/ct/totalsegmentator")
    TOTALSEG_REPO = os.path.abspath("external_repos/TotalSegmentator")
    if TOTALSEG_REPO not in sys.path:
        sys.path.insert(0, TOTALSEG_REPO)

    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map

    INPUT_CT = os.path.abspath("external_repos/TotalSegmentator/tests/reference_files/example_ct_sm.nii.gz")
    OUTPUT_DIR = os.path.abspath("outputs/ct_tests")
    OUTPUT_SEG = os.path.join(OUTPUT_DIR, "output_multilabel_3mm.nii.gz")
    REFERENCE_SEG = os.path.abspath("external_repos/TotalSegmentator/tests/reference_files/example_seg_fast.nii.gz")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Verification] Input CT scan: {INPUT_CT}")
    print(f"[Verification] Output target: {OUTPUT_SEG}")
    print(f"[Verification] Executing genuine TotalSegmentator inference on CPU (Task 297, fast=True, ml=True)...")

    start_time = time.time()

    # Execute real inference with nr_thr_resamp=1 and nr_thr_saving=1 for safe Windows execution
    res_img = totalsegmentator(
        input=INPUT_CT,
        output=OUTPUT_SEG,
        fast=True,
        ml=True,
        device="cpu",
        nr_thr_resamp=1,
        nr_thr_saving=1,
        quiet=False,
        verbose=False
    )

    elapsed = time.time() - start_time
    print(f"\n[Verification] Genuine inference completed in {elapsed:.2f} seconds!")

    # Verify output file exists on disk
    assert os.path.exists(OUTPUT_SEG), f"Output file not generated at {OUTPUT_SEG}"

    # Load generated segmentation NIfTI
    out_nib = nib.load(OUTPUT_SEG)
    out_data = out_nib.get_fdata()
    in_nib = nib.load(INPUT_CT)

    print(f"[Verification] Output NIfTI dimensions: {out_nib.shape}")
    print(f"[Verification] Input CT dimensions: {in_nib.shape}")
    assert out_nib.shape == in_nib.shape, f"Dimension mismatch: {out_nib.shape} vs {in_nib.shape}"

    # Check affine consistency
    np.testing.assert_allclose(out_nib.affine, in_nib.affine, rtol=1e-3, atol=1e-3)
    print(f"[Verification] Affine matrix and orientation match input CT perfectly!")

    # Analyze identified classes
    unique_labels = [int(x) for x in np.unique(out_data) if x > 0]
    print(f"[Verification] Identified {len(unique_labels)} anatomical structures in scan:")
    classes_117 = class_map.get("total_v1", {})

    for lbl in unique_labels[:15]:
        name = classes_117.get(lbl, f"class_{lbl}")
        count = int(np.sum(out_data == lbl))
        print(f"  - Label {lbl:3d} ({name:25s}): {count:6d} voxels")

    # Compare against reference ground truth
    if os.path.exists(REFERENCE_SEG):
        ref_nib = nib.load(REFERENCE_SEG)
        ref_data = ref_nib.get_fdata()
        intersection = np.sum((out_data > 0) & (ref_data > 0))
        total_fg = np.sum(out_data > 0) + np.sum(ref_data > 0)
        dice_overall = (2.0 * intersection) / (total_fg + 1e-8)
        print(f"[Verification] Agreement (Overall Foreground Dice) against upstream reference: {dice_overall:.4f}")

    print("\n[Verification] Real inference verified successfully with 0 errors!")

if __name__ == "__main__":
    main()
