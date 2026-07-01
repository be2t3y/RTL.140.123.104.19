#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect SPICE netlists from each Sram*/SPICE folder and create symbolic links in spi/.

Expected directory layout (under --root):

    <root>/
      Sram_16384/
        SPICE/  *.spi
      Sram_tok1/
        SPICE/  *.spi
      ...

After running, this creates under --root:

    <root>/spi/   <- symbolic links to all Sram*/SPICE/*.spi (pointing to source absolute path)

Default behavior:
  - Only process folders whose name starts with --prefix (default "Sram").
  - Subfolder name matching is case-insensitive (SPICE/spice).
  - File extension matching is case-insensitive (.spi/.SPI).
  - Create symbolic links; the link target is always the source file's absolute path.
  - If a link/file with the same name already exists at the destination, skip it
    (use --overwrite to replace).

Usage examples:
    python3 collect_spi.py                 # create symlinks in the current directory
    python3 collect_spi.py --root /path/to/Memory2
    python3 collect_spi.py --overwrite     # overwrite existing link/file with the same name
"""

import argparse
import os
import sys


def find_subdir(parent, target_name):
    """Find a subfolder under parent whose name equals target_name (case-insensitive)."""
    target_lower = target_name.lower()
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    for name in entries:
        full = os.path.join(parent, name)
        if os.path.isdir(full) and name.lower() == target_lower:
            return full
    return None


def list_files_with_ext(folder, ext):
    """List full paths of files in folder with extension ext (case-insensitive, e.g. ".spi")."""
    ext_lower = ext.lower()
    result = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() == ext_lower:
            result.append(full)
    return result


def transfer(src, dst_dir, overwrite=False):
    """Create a symbolic link in dst_dir pointing to the absolute path of src.

    Returns ("linked"/"skipped", dst_path).
    """
    src_abs = os.path.abspath(src)
    dst = os.path.join(dst_dir, os.path.basename(src))
    # os.path.exists returns False for a symlink pointing to a missing target, so use lexists
    if os.path.lexists(dst):
        if not overwrite:
            return ("skipped", dst)
        os.remove(dst)
    os.symlink(src_abs, dst)
    return ("linked", dst)


def collect(root, prefix, spi_subdir, out_spi, overwrite=False):
    out_spi_dir = os.path.join(root, out_spi)
    os.makedirs(out_spi_dir, exist_ok=True)

    prefix_lower = prefix.lower()
    sram_dirs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.lower().startswith(prefix_lower):
            # avoid scanning the output folder itself
            if os.path.abspath(full) == os.path.abspath(out_spi_dir):
                continue
            sram_dirs.append(full)

    if not sram_dirs:
        print("[WARN] no folder starting with '%s' found under %s" % (prefix, root))
        return

    stats = {"spi": 0, "spi_skip": 0, "no_spi": 0}

    for sram in sram_dirs:
        sram_name = os.path.basename(sram)

        spi_dir = find_subdir(sram, spi_subdir)
        if spi_dir is None:
            print("[--] %-40s no %s/ subfolder" % (sram_name, spi_subdir))
            stats["no_spi"] += 1
            continue

        spi_files = list_files_with_ext(spi_dir, ".spi")
        if not spi_files:
            print("[--] %-40s no *.spi in %s/" % (sram_name, spi_subdir))
        for f in spi_files:
            action, dst = transfer(f, out_spi_dir, overwrite)
            if action == "skipped":
                stats["spi_skip"] += 1
                print("[skip] %s already exists, skipping %s" % (os.path.basename(dst), sram_name))
            else:
                stats["spi"] += 1
                print("[spi ] %s -> %s" % (os.path.basename(dst), os.path.abspath(f)))

    print("\n===== done (symbolic link) =====")
    print("folders scanned   : %d" % len(sram_dirs))
    print("SPI links         : %d (skipped %d) -> %s" % (stats["spi"], stats["spi_skip"], out_spi_dir))
    if stats["no_spi"]:
        print("missing SPICE dir : %d" % stats["no_spi"])


def main():
    parser = argparse.ArgumentParser(
        description="Create symbolic links (absolute path) in spi/ for each Sram* folder's SPICE/*.spi")
    parser.add_argument("--root", default=".",
                        help="Root directory containing the Sram* folders (default: current directory)")
    parser.add_argument("--prefix", default="Sram",
                        help="Folder name prefix to process (default: Sram)")
    parser.add_argument("--spi-subdir", default="SPICE",
                        help="Subfolder name holding SPICE files (default: SPICE)")
    parser.add_argument("--out-spi", default="spi",
                        help="Output folder name for SPICE (default: spi)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite when a link/file with the same name exists (default: skip)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print("[ERROR] root is not a directory: %s" % root)
        sys.exit(1)

    collect(root, args.prefix, args.spi_subdir, args.out_spi, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
