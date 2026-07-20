"""
generate_phase16_stats.py

Phase 1.6 — Dataset Statistics Report Generator.

Reads qas.zip directly (no images or graphs needed) and produces:
  phase1_6_results/dataset_statistics.md

Usage
-----
# Local (only qas.zip available):
python generate_phase16_stats.py

# Workstation (full dataset path):
python generate_phase16_stats.py \\
    --data-root /home/cvpruts/Downloads/HDGT-main\\(1\\)/HDGT-main/data/MP-DocVQA
"""

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="HDGT Phase 1.6 — Generate MP-DocVQA dataset statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data/MP-DocVQA")
    p.add_argument("--output-dir", default="phase1_6_results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(root: Path, split: str) -> list:
    """Load a split from extracted JSON or directly from the ZIP."""
    json_path = root / f"{split}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("data", [])

    zip_path = root / "qas.zip"
    if not zip_path.exists():
        return []
    with zipfile.ZipFile(zip_path, "r") as zf:
        fname = f"{split}.json"
        if fname not in zf.namelist():
            return []
        with zf.open(fname) as f:
            return json.load(f).get("data", [])


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(all_items: list) -> dict:
    """Compute comprehensive dataset statistics from QA items."""

    split_counts      = Counter()
    doc_ids           = set()
    unique_contexts   = {}
    page_counts       = []           # pages per context
    question_lengths  = []           # word count
    answer_lengths    = []           # char count of first answer
    multi_answer      = 0
    single_page_ctx   = 0
    multi_page_ctx    = 0
    answer_page_dist  = Counter()    # which page index has the answer

    for item in all_items:
        split  = item.get("data_split", "unknown")
        doc_id = item.get("doc_id", "")
        pages  = item.get("page_ids", [])
        ans    = item.get("answers", [])
        ap_idx = item.get("answer_page_idx", None)

        split_counts[split] += 1
        doc_ids.add(doc_id)

        # Build context_id
        pnums = []
        for pid in pages:
            m = re.search(r"_p(\d+)$", pid)
            pnums.append(int(m.group(1)) if m else 0)
        pnums.sort()
        cid = f"{doc_id}_p{min(pnums, default=0)}_p{max(pnums, default=0)}"
        if cid not in unique_contexts:
            unique_contexts[cid] = len(pages)
            page_counts.append(len(pages))
            if len(pages) == 1:
                single_page_ctx += 1
            else:
                multi_page_ctx += 1

        # Question stats
        q = item.get("question", "")
        question_lengths.append(len(q.split()))

        # Answer stats
        if ans:
            answer_lengths.append(len(ans[0]))
        if len(ans) > 1:
            multi_answer += 1

        # Answer page distribution
        if ap_idx is not None:
            answer_page_dist[ap_idx] += 1

    return {
        "total_questions": len(all_items),
        "split_counts":    dict(split_counts),
        "unique_docs":     len(doc_ids),
        "unique_contexts": len(unique_contexts),
        "page_counts":     page_counts,
        "question_lengths": question_lengths,
        "answer_lengths":  answer_lengths,
        "multi_answer":    multi_answer,
        "single_page_ctx": single_page_ctx,
        "multi_page_ctx":  multi_page_ctx,
        "answer_page_dist": dict(answer_page_dist),
    }


def _avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def _median(lst):
    if not lst:
        return 0
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(stats: dict) -> str:
    pc = stats["page_counts"]
    ql = stats["question_lengths"]
    al = stats["answer_lengths"]

    # Answer page distribution (top 6)
    apd = sorted(stats["answer_page_dist"].items())[:6]
    apd_rows = "\n".join(
        f"| {idx:>10} | {cnt:>10,} |" for idx, cnt in apd
    )

    report = f"""# HDGT Phase 1.6 — MP-DocVQA Dataset Statistics Report

**Generated by**: `generate_phase16_stats.py`
**Dataset**: MP-DocVQA v1.0
**Source**: https://rrc.cvc.uab.es/?ch=17

---

## 1. Overview

| Property                         | Value           |
|----------------------------------|-----------------|
| Total QA pairs                   | {stats["total_questions"]:>15,} |
| Unique documents                 | {stats["unique_docs"]:>15,} |
| Unique page-range contexts       | {stats["unique_contexts"]:>15,} |
| Questions with multiple answers  | {stats["multi_answer"]:>15,} |

---

## 2. Split Distribution

| Split  | Questions |
|--------|----------:|
| Train  | {stats["split_counts"].get("train", 0):>9,} |
| Val    | {stats["split_counts"].get("val",   0):>9,} |
| Test   | {stats["split_counts"].get("test",  0):>9,} |
| **Total** | **{stats["total_questions"]:>7,}** |

---

## 3. Pages per Context

| Metric                      | Value   |
|-----------------------------|---------|
| Mean pages per context      | {_avg(pc):.2f}  |
| Median pages per context    | {_median(pc):.1f}  |
| Min pages                   | {min(pc, default=0)}       |
| Max pages                   | {max(pc, default=0)}      |
| Single-page contexts        | {stats["single_page_ctx"]:,}  |
| Multi-page contexts         | {stats["multi_page_ctx"]:,}  |
| % multi-page                | {100*stats["multi_page_ctx"]/max(1, stats["unique_contexts"]):.1f}%   |

> [!NOTE]
> The MP-DocVQA challenge caps document length at 20 pages.
> Multi-page contexts require cross-page reasoning — the primary use case
> for HDGT's graph-based retrieval over flat page-level methods.

---

## 4. Question & Answer Statistics

| Metric                        | Value   |
|-------------------------------|---------|
| Mean question length (words)  | {_avg(ql):.1f}   |
| Median question length        | {_median(ql):.1f}   |
| Mean answer length (chars)    | {_avg(al):.1f}   |
| Median answer length (chars)  | {_median(al):.1f}   |

---

## 5. Answer Page Distribution

The `answer_page_idx` field is 0-indexed within the context page list.
A value of 0 means the answer is on the first page of the context.

| answer_page_idx | Questions  |
|----------------:|-----------:|
{apd_rows}

> [!NOTE]
> High concentration at low page indices is expected — most questions about
> a document are answered in the first few pages of their context window.
> Multi-hop questions where the answer page is high (e.g. index ≥ 10) are
> exactly where HDGT's graph traversal provides the greatest advantage.

---

## 6. Implications for HDGT Evaluation

### Retrieval granularity
Based on the page distribution above, retrieval must work at the **node level**
within pages — not just at the page level — to support Evidence Localization
Accuracy (ELA) measurement.

### Evaluation split
- **Primary**: `val` split ({stats["split_counts"].get("val", 0):,} questions,
  answers available).
- **Secondary**: `train` subset for ablation studies.
- **Withheld**: `test` split (no answers — requires submission to the challenge
  server for official ANLS scoring).

### Next steps
1. Run `python prepare_mpdocvqa.py --data-root <path>` to extract zips and
   compile multi-page PDFs.
2. Run `python build_mpdocvqa_graphs.py --data-root <path>` to build graphs.
3. Run `python evaluate_retrieval.py --split val --method bm25` to evaluate.
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    root_dir = Path(args.data_root).expanduser().resolve()
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HDGT Phase 1.6 — Dataset Statistics Generator")
    print("=" * 60)
    print(f"  Data root  : {root_dir}")

    # Load all splits
    all_items = []
    for split in ["train", "val", "test"]:
        items = load_split(root_dir, split)
        print(f"  Loaded {len(items):>6,} items from {split}")
        all_items.extend(items)

    if not all_items:
        print(
            "\n[ERROR] No QA data found.\n"
            f"  Expected qas.zip or extracted JSONs in: {root_dir}"
        )
        sys.exit(1)

    print(f"\n  Computing statistics over {len(all_items):,} QA items...")
    stats  = compute_stats(all_items)
    report = generate_report(stats)

    out_path = out_dir / "dataset_statistics.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Report saved: {out_path}")
    print("=" * 60)
    print(f"  Total QA pairs     : {stats['total_questions']:,}")
    print(f"  Unique documents   : {stats['unique_docs']:,}")
    print(f"  Unique contexts    : {stats['unique_contexts']:,}")
    print(f"  Avg pages/context  : {_avg(stats['page_counts']):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
