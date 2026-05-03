"""
准备 KITTI 深度评估数据集。

功能：
1. 从 data_depth_annotated.zip 解压深度真值（去掉 train/val 前缀）
2. 从 KITTI Raw 创建 RGB 图像的软链接
3. 生成过滤后的 eigen 文件列表（仅保留实际存在的样本）
4. 创建 Marigold 配置所需的目录结构

最终目录结构 (/home/chenfan/Datasets/preparedAll/kitti/):
  ├── 2011_09_26/                            ← RGB (软链接到 Raw KITTI)
  │   └── 2011_09_26_drive_XXXX_sync/
  │       └── image_02/data/*.png
  ├── 2011_09_26_drive_XXXX_sync/            ← Depth GT (从 zip 解压)
  │   └── proj_depth/groundtruth/image_02/*.png
  ...

用法:
  python prepare_kitti.py [--dry-run]
"""

import argparse
import os
import sys
import zipfile
from collections import defaultdict

# ==================== 配置 ====================
KITTI_RAW_DIR = "/media/chenfan/chenfan/datasets/Kitti/kitti"
DEPTH_ZIP = "/media/chenfan/chenfan/datasets/Kitti/kitti真值/data_depth_annotated.zip"
OUTPUT_DIR = "/home/chenfan/Datasets/preparedAll/kitti"
SPLIT_DIR = "/home/chenfan/projectsVScode/CMarigold/Marigold-main/data_split/kitti_depth"

SPLIT_FILES = {
    "eigen_test_files_with_gt.txt": "eigen_test_files_with_gt_filtered.txt",
    "eigen_val_from_train_800.txt": "eigen_val_from_train_800_filtered.txt",
    "eigen_val_from_train_sub_100.txt": "eigen_val_from_train_sub_100_filtered.txt",
}
# ===============================================


def parse_file_list(filepath):
    """解析 eigen 文件列表，返回 [(rgb_path, depth_path, focal), ...]"""
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                entries.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                entries.append((parts[0], parts[1], ""))
    return entries


def get_date_from_drive(drive_name):
    """从 drive 名提取日期: 2011_09_26_drive_0002_sync -> 2011_09_26"""
    parts = drive_name.split("_drive_")
    return parts[0] if parts else None


def main():
    parser = argparse.ArgumentParser(description="准备 KITTI 深度评估数据")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不实际操作")
    args = parser.parse_args()

    # ---- 检查输入 ----
    print("=" * 60)
    print("KITTI 深度评估数据准备脚本")
    print("=" * 60)

    if not os.path.isdir(KITTI_RAW_DIR):
        print(f"[ERROR] KITTI Raw 目录不存在: {KITTI_RAW_DIR}")
        sys.exit(1)
    if not os.path.isfile(DEPTH_ZIP):
        print(f"[ERROR] 深度真值 zip 不存在: {DEPTH_ZIP}")
        sys.exit(1)

    print(f"[OK] KITTI Raw:  {KITTI_RAW_DIR}")
    print(f"[OK] Depth zip:  {DEPTH_ZIP}")
    print(f"[OK] 输出目录:   {OUTPUT_DIR}")
    print()

    # ---- Step 1: 收集所有需要的 date/drive ----
    print("Step 1: 分析文件列表...")

    all_entries = {}  # filename -> [(rgb, depth, focal), ...]
    needed_dates = set()  # 需要的日期目录 (RGB)
    needed_drives = set()  # 需要的 drive 目录 (Depth)

    for src_file in SPLIT_FILES:
        src_path = os.path.join(SPLIT_DIR, src_file)
        if not os.path.exists(src_path):
            print(f"  [WARN] 文件列表不存在: {src_path}")
            continue
        entries = parse_file_list(src_path)
        all_entries[src_file] = entries

        for rgb_path, depth_path, _ in entries:
            # RGB: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/xxx.png
            date = rgb_path.split("/")[0]
            needed_dates.add(date)
            # Depth: 2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/xxx.png
            drive = depth_path.split("/")[0]
            needed_drives.add(drive)

    print(f"  需要 {len(needed_dates)} 个日期目录 (RGB)")
    print(f"  需要 {len(needed_drives)} 个 drive 目录 (Depth)")
    print()

    # ---- Step 2: 检查 Raw KITTI 中哪些存在 ----
    print("Step 2: 检查 KITTI Raw 数据覆盖...")
    available_dates = set()
    available_drives = set()

    for date in sorted(needed_dates):
        date_path = os.path.join(KITTI_RAW_DIR, date)
        if os.path.isdir(date_path):
            available_dates.add(date)
            # 检查该日期下的 drive
            for drive in os.listdir(date_path):
                drive_path = os.path.join(date_path, drive)
                if os.path.isdir(drive_path) and "drive" in drive:
                    available_drives.add(drive)
        else:
            print(f"  [MISS] 日期目录缺失: {date}")

    missing_drives = needed_drives - available_drives
    if missing_drives:
        print(f"  [WARN] 缺少 {len(missing_drives)} 个 drive 目录:")
        for d in sorted(missing_drives):
            print(f"         {d}")
    print(f"  RGB 覆盖: {len(available_drives)}/{len(needed_drives)} drives")
    print()

    # ---- Step 3: 检查 zip 中的深度数据 ----
    print("Step 3: 检查深度真值 zip...")
    zip_depth_files = set()
    with zipfile.ZipFile(DEPTH_ZIP, "r") as z:
        for name in z.namelist():
            if name.endswith(".png"):
                # 去掉 train/ 或 val/ 前缀
                parts = name.split("/", 1)
                if len(parts) == 2 and parts[0] in ("train", "val"):
                    zip_depth_files.add(parts[1])

    available_depth = needed_drives.intersection(
        {f.split("/")[0] for f in zip_depth_files}
    )
    print(f"  Zip 中包含 {len(zip_depth_files)} 个深度 PNG")
    print(f"  Depth 覆盖: {len(available_depth)}/{len(needed_drives)} drives")
    print()

    # ---- Step 4: 生成过滤后的文件列表 ----
    print("Step 4: 生成过滤后的文件列表...")
    for src_file, dst_file in SPLIT_FILES.items():
        if src_file not in all_entries:
            continue

        entries = all_entries[src_file]
        filtered = []
        skipped = 0

        for rgb_path, depth_path, focal in entries:
            # 检查 RGB 是否存在
            rgb_full = os.path.join(KITTI_RAW_DIR, rgb_path)
            rgb_ok = os.path.exists(rgb_full)

            # 检查 depth 是否在 zip 中
            depth_ok = depth_path in zip_depth_files

            if rgb_ok and depth_ok:
                if focal:
                    filtered.append(f"{rgb_path} {depth_path} {focal}")
                else:
                    filtered.append(f"{rgb_path} {depth_path}")
            else:
                skipped += 1

        dst_path = os.path.join(SPLIT_DIR, dst_file)
        print(f"  {src_file}:")
        print(f"    原始: {len(entries)}, 保留: {len(filtered)}, 跳过: {skipped}")

        if not args.dry_run:
            with open(dst_path, "w") as f:
                f.write("\n".join(filtered) + "\n")
            print(f"    写入: {dst_path}")
        else:
            print(f"    [dry-run] 将写入: {dst_path}")
    print()

    if args.dry_run:
        print("[dry-run] 以下操作将在实际运行时执行：")
        print(f"  - 创建输出目录: {OUTPUT_DIR}")
        print(f"  - 创建 {len(needed_dates)} 个日期目录软链接 (RGB)")
        print(f"  - 从 zip 解压深度真值到 {OUTPUT_DIR}")
        print("重新运行不带 --dry-run 以执行。")
        return

    # ---- Step 5: 创建输出目录 ----
    print("Step 5: 创建输出目录结构...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 6: 创建 RGB 软链接 ----
    print("Step 6: 创建 RGB 软链接...")
    for date in sorted(needed_dates):
        src = os.path.join(KITTI_RAW_DIR, date)
        dst = os.path.join(OUTPUT_DIR, date)
        if os.path.exists(dst):
            if os.path.islink(dst):
                current_target = os.readlink(dst)
                if current_target == src:
                    print(f"  [SKIP] {date} → 已存在且正确")
                    continue
                else:
                    os.remove(dst)
            else:
                print(f"  [WARN] {dst} 已存在但不是软链接，跳过")
                continue

        if os.path.isdir(src):
            os.symlink(src, dst)
            print(f"  [LINK] {date} → {src}")
        else:
            print(f"  [MISS] {date} 不存在于 Raw KITTI，跳过")
    print()

    # ---- Step 7: 解压深度真值 ----
    print("Step 7: 解压深度真值 (去掉 train/val 前缀)...")
    extracted = 0
    skipped_extract = 0

    with zipfile.ZipFile(DEPTH_ZIP, "r") as z:
        members = [m for m in z.infolist() if m.filename.endswith(".png")]
        total = len(members)

        for i, member in enumerate(members):
            # 去掉 train/ 或 val/ 前缀
            parts = member.filename.split("/", 1)
            if len(parts) != 2 or parts[0] not in ("train", "val"):
                continue

            rel_path = parts[1]  # e.g. 2011_09_26_drive_0001_sync/proj_depth/...
            dst_path = os.path.join(OUTPUT_DIR, rel_path)

            if os.path.exists(dst_path):
                skipped_extract += 1
            else:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with z.open(member) as src_f, open(dst_path, "wb") as dst_f:
                    dst_f.write(src_f.read())
                extracted += 1

            if (i + 1) % 5000 == 0 or (i + 1) == total:
                print(
                    f"\r  进度: {i+1}/{total}  "
                    f"(解压: {extracted}, 跳过: {skipped_extract})",
                    end="",
                    flush=True,
                )

    print(f"\n  解压完成: 新增 {extracted} 文件, 跳过 {skipped_extract} 已存在文件")
    print()

    # ---- Step 8: 验证 ----
    print("Step 8: 最终验证...")
    for src_file, dst_file in SPLIT_FILES.items():
        dst_path = os.path.join(SPLIT_DIR, dst_file)
        if not os.path.exists(dst_path):
            continue

        with open(dst_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        ok = 0
        fail = 0
        for line in lines:
            parts = line.split()
            rgb_full = os.path.join(OUTPUT_DIR, parts[0])
            depth_full = os.path.join(OUTPUT_DIR, parts[1])
            if os.path.exists(rgb_full) and os.path.exists(depth_full):
                ok += 1
            else:
                fail += 1

        status = "✓" if fail == 0 else "✗"
        print(f"  {status} {dst_file}: {ok} OK, {fail} FAIL")

    # ---- 完成 ----
    print()
    print("=" * 60)
    print("完成！")
    print(f"KITTI 数据目录: {OUTPUT_DIR}")
    print()
    print("Marigold 配置中使用:")
    print(f'  dir: preparedAll/kitti')
    print(f'  BASE_DATA_DIR: /home/chenfan/Datasets')
    print()
    print("过滤后的文件列表:")
    for _, dst_file in SPLIT_FILES.items():
        print(f"  data_split/kitti_depth/{dst_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
