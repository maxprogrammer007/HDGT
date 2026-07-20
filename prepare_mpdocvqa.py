"""
prepare_mpdocvqa.py

Phase 1.6 — MP-DocVQA Dataset Preparation Script.

This script:
  1. Auto-extracts all available zip files (qas, images, ocr, imdbs).
  2. Reads the QA JSON splits and identifies unique page-range contexts.
  3. Compiles page images for each context into a multi-page PDF using Pillow.
  4. Saves a context_map.json mapping question IDs to their PDF paths.

Usage
-----
# On the workstation (full dataset path):
python prepare_mpdocvqa.py \\
    --data-root /home/cvpruts/Downloads/HDGT-main\\(1\\)/HDGT-main/data/MP-DocVQA

# On local machine (only qas.zip present — skips PDF compilation):
python prepare_mpdocvqa.py --data-root data/MP-DocVQA

Expected zip files in --data-root:
  qas.zip    — Questions and Answers (train/val/test JSON)
  images.zip — Page images (.png per page)
  ocr.zip    — Amazon Textract OCR results (optional, not used by HDGT graph builder)
  imdbs.zip  — Processed IMDBs for the MP-DocVQA framework (optional)
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDGT — MP-DocVQA Dataset Preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="data/MP-DocVQA",
        help=(
            "Path to the MP-DocVQA directory containing the zip files. "
            "Example (workstation): "
            "/home/cvpruts/Downloads/HDGT-main(1)/HDGT-main/data/MP-DocVQA"
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip zip extraction even if zip files are present.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip PDF compilation (useful if only inspecting QA stats).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------------

# Map from zip filename stem → subdirectory to extract into
ZIP_MAP = {
    "qas":    None,       # Extract train.json, val.json, test.json directly to root
    "images": "images",   # Extract page images to images/
    "ocr":    "ocr",      # Amazon Textract OCR JSON files
    "imdbs":  "mpdocvqa_imdbs",  # Processed IMDB files
}


def extract_zips(root_dir: Path, skip: bool = False) -> None:
    """Auto-detect and extract all known zip files in root_dir."""
    print("\n[1/4] Checking and extracting zip files...")
    for stem, subdir in ZIP_MAP.items():
        zip_path = root_dir / f"{stem}.zip"
        if not zip_path.exists():
            print(f"  [SKIP] {stem}.zip — not found.")
            continue

        # Determine extraction target
        target = root_dir if subdir is None else root_dir / subdir

        # Check if already extracted (heuristic)
        already_extracted = False
        if stem == "qas":
            already_extracted = all(
                (root_dir / f"{s}.json").exists()
                for s in ["train", "val", "test"]
            )
        else:
            already_extracted = target.exists() and any(target.iterdir())

        if already_extracted:
            print(f"  [OK]   {stem}.zip — already extracted.")
            continue

        if skip:
            print(f"  [SKIP] {stem}.zip — --skip-extract flag set.")
            continue

        print(f"  Extracting {stem}.zip → {target} ...")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract with progress bar
            members = zf.infolist()
            for member in tqdm(members, desc=f"  {stem}.zip", unit="file"):
                zf.extract(member, target)
        print(f"  [DONE] {stem}.zip extracted ({len(members)} files).")


# ---------------------------------------------------------------------------
# Context identification
# ---------------------------------------------------------------------------

def identify_contexts(root_dir: Path):
    """
    Read QA split JSONs and derive unique (doc_id, page_range) contexts.

    Returns
    -------
    unique_contexts : dict  context_id → {doc_id, page_ids, split}
    questions_map   : dict  question_id → {context_id, answer_page_idx, answers}
    """
    print("\n[2/4] Reading QA splits and identifying contexts...")

    unique_contexts = {}
    questions_map   = {}

    for split in ["train", "val", "test"]:
        split_file = root_dir / f"{split}.json"
        if not split_file.exists():
            print(f"  [SKIP] {split}.json — not found.")
            continue

        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("data", [])
        for item in tqdm(items, desc=f"  {split}", unit="q"):
            q_id    = str(item["questionId"])
            doc_id  = item["doc_id"]
            page_ids = item["page_ids"]

            # Parse page numbers and sort
            pnums = []
            for pid in page_ids:
                m = re.search(r"_p(\d+)$", pid)
                pnums.append(int(m.group(1)) if m else 0)

            sorted_pairs   = sorted(zip(pnums, page_ids))
            sorted_page_ids = [pid for _, pid in sorted_pairs]
            sorted_pnums    = [pn  for pn,  _ in sorted_pairs]

            start_p    = sorted_pnums[0]  if sorted_pnums else 0
            end_p      = sorted_pnums[-1] if sorted_pnums else 0
            context_id = f"{doc_id}_p{start_p}_p{end_p}"

            if context_id not in unique_contexts:
                unique_contexts[context_id] = {
                    "doc_id":   doc_id,
                    "page_ids": sorted_page_ids,
                    "split":    split,
                }

            questions_map[q_id] = {
                "context_id":      context_id,
                "answer_page_idx": item.get("answer_page_idx", None),
                "answers":         item.get("answers", []),
            }

    print(f"  Found {len(unique_contexts):,} unique page contexts.")
    print(f"  Found {len(questions_map):,} questions mapped.")
    return unique_contexts, questions_map


# ---------------------------------------------------------------------------
# PDF compilation
# ---------------------------------------------------------------------------

def compile_pdfs(root_dir: Path, unique_contexts: dict) -> tuple:
    """
    Compile page images for each context into a multi-page PDF using Pillow.

    Returns (compiled_count, missing_count).
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [ERROR] Pillow is not installed. Run: pip install pillow")
        return 0, len(unique_contexts)

    print("\n[3/4] Compiling page images into multi-page PDFs...")

    images_dir   = root_dir / "images"
    pdfs_dir     = root_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(
            f"  [ERROR] images/ directory not found at {images_dir}.\n"
            f"  Make sure images.zip has been extracted first.\n"
            f"  PDF compilation skipped."
        )
        return 0, len(unique_contexts)

    compiled_count = 0
    missing_count  = 0

    for context_id, ctx in tqdm(unique_contexts.items(), desc="  Compiling PDFs"):
        pdf_path = pdfs_dir / f"{context_id}.pdf"
        if pdf_path.exists():
            compiled_count += 1
            continue

        # Locate image files for each page
        img_paths = []
        missing = False
        for pid in ctx["page_ids"]:
            found = None
            for ext in (".png", ".jpg", ".jpeg"):
                p = images_dir / f"{pid}{ext}"
                if p.exists():
                    found = p
                    break
            if found is None:
                # Recursive search (some datasets nest images in subfolders)
                candidates = list(images_dir.glob(f"**/{pid}.*"))
                if candidates:
                    found = candidates[0]
            if found is None:
                missing = True
                break
            img_paths.append(found)

        if missing:
            missing_count += 1
            continue

        # Compile with Pillow
        try:
            pil_images = [Image.open(p).convert("RGB") for p in img_paths]
            pil_images[0].save(
                pdf_path,
                save_all=True,
                append_images=pil_images[1:],
            )
            compiled_count += 1
        except Exception as e:
            print(f"\n  [ERROR] Failed to compile {context_id}: {e}")
            missing_count += 1

    return compiled_count, missing_count


# ---------------------------------------------------------------------------
# Save mapping
# ---------------------------------------------------------------------------

def save_mapping(root_dir: Path, unique_contexts: dict, questions_map: dict) -> Path:
    """Save context_map.json for use by build_mpdocvqa_graphs.py."""
    print("\n[4/4] Saving context_map.json...")
    mapping = {"contexts": unique_contexts, "questions": questions_map}
    mapping_path = root_dir / "context_map.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {mapping_path}")
    return mapping_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args     = parse_args()
    root_dir = Path(args.data_root).expanduser().resolve()

    print("=" * 60)
    print("  HDGT — MP-DocVQA Dataset Preparation")
    print("=" * 60)
    print(f"  Data root : {root_dir}")

    if not root_dir.exists():
        print(f"\n[ERROR] Data root does not exist: {root_dir}")
        sys.exit(1)

    # Step 1: Extract zips
    extract_zips(root_dir, skip=args.skip_extract)

    # Step 2: Identify unique contexts from QA JSONs
    unique_contexts, questions_map = identify_contexts(root_dir)

    if not unique_contexts:
        print(
            "\n[ERROR] No contexts found. Make sure qas.zip has been extracted "
            "and train.json / val.json / test.json are present in the data root."
        )
        sys.exit(1)

    # Step 3: Compile PDFs (only if images are available)
    if args.skip_compile:
        print("\n[3/4] PDF compilation skipped (--skip-compile).")
        compiled_count = missing_count = 0
    else:
        compiled_count, missing_count = compile_pdfs(root_dir, unique_contexts)

    # Step 4: Save mapping
    save_mapping(root_dir, unique_contexts, questions_map)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Total contexts identified : {len(unique_contexts):,}")
    print(f"  Total questions mapped    : {len(questions_map):,}")
    if not args.skip_compile:
        print(f"  PDFs compiled             : {compiled_count:,}")
        if missing_count > 0:
            print(f"  PDFs skipped (missing img): {missing_count:,}")
    print("=" * 60)
    print("\nNext step on workstation:")
    print(
        f"  python build_mpdocvqa_graphs.py "
        f"--data-root {args.data_root}"
    )


if __name__ == "__main__":
    main()
