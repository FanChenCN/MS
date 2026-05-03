#
# Hypersim subset download script
# Strategy: 1 scene per group (50 groups) for maximum scene diversity
# Estimated download size: ~200 GB (50 zips x ~4 GB each)
# Run with: python download_hypersim_subset.py --downloads_dir /path/to/downloads --decompress_dir /path/to/raw --delete_archive_after_decompress
#

import argparse
import os
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--downloads_dir", required=True)
parser.add_argument("--decompress_dir")
parser.add_argument("--delete_archive_after_decompress", action="store_true")
args = parser.parse_args()

os.makedirs(args.downloads_dir, exist_ok=True)
if args.decompress_dir:
    os.makedirs(args.decompress_dir, exist_ok=True)

def download(url):
    name = os.path.basename(url)
    dest = os.path.join(args.downloads_dir, name)
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0

    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"} if existing else {})
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(dest, "ab" if existing else "wb") as f:
                total = existing + int(resp.headers.get("Content-Length", 0))
                downloaded = existing
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\r  {name}: {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB", end="", flush=True)
            print()
            break
        except Exception as e:
            print(f"\n  [attempt {attempt}/5 failed] {e}")
            if attempt < 5:
                time.sleep(10)
                existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            else:
                print(f"  [SKIP] {name} — will retry on next run")
                return False

    if args.decompress_dir:
        import zipfile
        print(f"  Extracting {name}...")
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(args.decompress_dir)
        if args.delete_archive_after_decompress:
            os.remove(dest)
    return True

# One scene per group — 50 groups, maximum diversity
urls_to_download = [
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_014_003.zip",  # group 014 starts at _003
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_001.zip",  # no ai_020
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_001.zip",  # no ai_025
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_029_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_034_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_002.zip",  # group 038 starts at _002
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_002.zip",  # group 039 starts at _002
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_001.zip",  # no ai_040
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_042_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_002.zip",  # group 043 starts at _002
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_001.zip",  # no ai_049
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_051_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_001.zip",
]

print(f"[HYPERSIM SUBSET] Downloading {len(urls_to_download)} scenes (1 per group)...")
failed = []
for i, url in enumerate(urls_to_download, 1):
    name = os.path.basename(url)
    dest = os.path.join(args.downloads_dir, name)
    # Skip already completed downloads (zip should be valid)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
        try:
            import zipfile
            with zipfile.ZipFile(dest, "r") as z:
                z.testzip()
            print(f"[{i}/{len(urls_to_download)}] {name} already complete, skip")
            continue
        except Exception:
            pass  # corrupted, re-download
    print(f"[{i}/{len(urls_to_download)}] {name}")
    if not download(url):
        failed.append(name)

if failed:
    print(f"\n[WARNING] {len(failed)} files failed:")
    for f in failed:
        print(f"  - {f}")
    print("Re-run the script to retry failed downloads.")
else:
    print("[HYPERSIM SUBSET] All done!")
