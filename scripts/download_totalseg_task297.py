"""
Download and unpack TotalSegmentator Fast 3mm Task 297 Checkpoint (129.11 MB).
Official upstream source: https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip
"""

import os
import sys
import zipfile
import urllib.request
import time

URL = "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip"
TARGET_DIR = os.path.abspath("checkpoints/ct/totalsegmentator")
ZIP_PATH = os.path.join(TARGET_DIR, "Dataset297_TotalSegmentator_total_3mm_1559subj.zip")
UNPACKED_DIR = os.path.join(TARGET_DIR, "Dataset297_TotalSegmentator_total_3mm_1559subj")

os.makedirs(TARGET_DIR, exist_ok=True)

if os.path.exists(UNPACKED_DIR) and len(os.listdir(UNPACKED_DIR)) > 0:
    print(f"[Download] Pretrained model directory already exists at: {UNPACKED_DIR}")
    sys.exit(0)

print(f"[Download] Starting download of TotalSegmentator Fast 3mm model (Task 297)...")
print(f"[Download] Source URL: {URL}")
print(f"[Download] Destination: {ZIP_PATH}")

start_time = time.time()

def report_progress(blocks_transferred, block_size, total_size):
    transferred_mb = (blocks_transferred * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    pct = (transferred_mb / total_mb) * 100 if total_mb > 0 else 0
    sys.stdout.write(f"\r[Download] {transferred_mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)")
    sys.stdout.flush()

urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=report_progress)
print(f"\n[Download] Download completed in {time.time() - start_time:.1f} seconds. Size: {os.path.getsize(ZIP_PATH) / 1024 / 1024:.2f} MB")

print(f"[Download] Extracting {ZIP_PATH} to {TARGET_DIR} ...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(TARGET_DIR)

print(f"[Download] Extraction finished. Removing temporary zip file...")
os.remove(ZIP_PATH)

print(f"[Download] Pretrained model ready at: {UNPACKED_DIR}")
for root, dirs, files in os.walk(UNPACKED_DIR):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), UNPACKED_DIR)
        sz = os.path.getsize(os.path.join(root, f)) / 1024 / 1024
        print(f"  - {rel} ({sz:.2f} MB)")
