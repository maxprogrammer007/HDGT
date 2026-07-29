"""
failure_analysis.py

HDGT Failure Analysis — Priority 4 from research review.

Loads the BM25 evaluation JSONL output, samples failed questions,
and categorises each failure into one of:
    - keyword_mismatch   : question terms not in retrieved content
    - ocr_error          : likely OCR artifact (numbers/symbols corrupted)
    - wrong_page         : correct page exists but was ranked too low
    - table_reasoning    : question involves numerical or table computation
    - visual_ambiguity   : answer depends on image layout/visual context
    - unknown            : does not fit above categories

Outputs:
    - experiments/failure_analysis.json   (per-failure records)
    - experiments/failure_summary.md      (markdown table for thesis)

Usage:
    python failure_analysis.py --method bm25 --n_failures 200
"""

import argparse
import json
import re
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.WARNING)

# -----------------------------------------------------------------------
# Heuristic classifiers
# -----------------------------------------------------------------------

TABLE_WORDS = {
    "total", "sum", "average", "percent", "how many", "how much",
    "count", "number of", "calculate", "compute", "amount",
    "add", "subtract", "multiply", "ratio", "rate",
}
VISUAL_WORDS = {
    "logo", "image", "photo", "picture", "signature", "stamp",
    "color", "colour", "diagram", "figure", "chart", "graph",
    "handwritten", "written", "printed",
}
OCR_PATTERN = re.compile(r"[#@|]{2,}|[0-9]{8,}|[A-Z]{5,}\d+|\b[A-Z0-9]{8,}\b")


def classify_failure(question: str, gt_answers: list, retrieved_contents: list,
                     gt_page: int, retrieved_pages: list) -> str:
    """
    Assign a failure category based on question text, retrieved content,
    and page-level retrieval outcome.
    """
    q_low = question.lower()
    top_content = " ".join(retrieved_contents[:3]).lower() if retrieved_contents else ""

    # --- wrong_page: retrieved correct page at rank > 1 ---
    if gt_page in retrieved_pages and retrieved_pages.index(gt_page) > 0:
        return "wrong_page"

    # --- table_reasoning: numerical or aggregation question ---
    if any(tw in q_low for tw in TABLE_WORDS):
        return "table_reasoning"

    # --- visual_ambiguity: answer depends on visual element ---
    if any(vw in q_low for vw in VISUAL_WORDS):
        return "visual_ambiguity"

    # --- ocr_error: answer string has unusual character patterns ---
    all_answers = " ".join(gt_answers)
    if OCR_PATTERN.search(all_answers):
        return "ocr_error"

    # --- keyword_mismatch: question keywords not in top retrieved text ---
    # Extract content-words from question (skip stop words)
    stop = {"what", "is", "the", "of", "in", "a", "an", "on", "at",
            "to", "for", "and", "or", "this", "that", "how", "which",
            "was", "are", "were", "has", "have", "been", "be", "by"}
    q_tokens = set(q_low.split()) - stop
    if q_tokens and top_content:
        overlap = sum(1 for t in q_tokens if t in top_content)
        if overlap == 0:
            return "keyword_mismatch"

    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="bm25", help="Evaluation method name used in JSONL filename")
    p.add_argument("--n_failures", type=int, default=300, help="Max number of failures to analyse")
    args = p.parse_args()

    jsonl_path = Path(f"experiments/retrieval_results_val_{args.method}.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No JSONL at {jsonl_path}. Run evaluate_retrieval.py first.")

    print(f"\nLoading results from {jsonl_path} ...")
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Identify failures (Recall@1 = 0)
    failures = [r for r in records if r["metrics"].get("recall@1", 0) == 0.0]
    successes = [r for r in records if r["metrics"].get("recall@1", 1) == 1.0]

    print(f"  Total questions : {len(records)}")
    print(f"  Recall@1 = 1    : {len(successes)} ({100*len(successes)/max(len(records),1):.1f}%)")
    print(f"  Recall@1 = 0    : {len(failures)} ({100*len(failures)/max(len(records),1):.1f}%)")

    # Sample up to n_failures
    sampled = failures[:args.n_failures]
    print(f"  Analysing      : {len(sampled)} failures\n")

    # Classify each failure
    categorised = []
    for r in sampled:
        question      = r.get("query", "")
        gt_answers    = r.get("ground_truth_answers", [])
        gt_page       = r.get("ground_truth_page", 0)
        ret_nodes     = r.get("retrieved_nodes", [])
        ret_pages     = [n.get("page", 0) for n in ret_nodes]
        ret_contents  = [n.get("content", "") for n in ret_nodes]

        cat = classify_failure(question, gt_answers, ret_contents, gt_page, ret_pages)
        categorised.append({
            "question_id": r.get("question_id"),
            "question":    question,
            "gt_page":     gt_page,
            "gt_answers":  gt_answers,
            "retrieved_pages": ret_pages,
            "top_content": ret_contents[0][:120] if ret_contents else "",
            "category":    cat,
            "mrr":         r["metrics"].get("mrr", 0),
            "ela":         r["metrics"].get("ela", 0),
        })

    counts = Counter(item["category"] for item in categorised)

    # Print summary table
    print("┌─────────────────────────────────┬───────┬─────────┐")
    print("│ Failure Category                │ Count │  Share  │")
    print("├─────────────────────────────────┼───────┼─────────┤")
    for cat, cnt in counts.most_common():
        pct = 100 * cnt / len(sampled)
        print(f"│ {cat:<31} │  {cnt:4d} │  {pct:5.1f}% │")
    print("├─────────────────────────────────┼───────┼─────────┤")
    print(f"│ {'TOTAL':<31} │  {len(sampled):4d} │  100.0% │")
    print("└─────────────────────────────────┴───────┴─────────┘")

    # Markdown summary for thesis
    md_path = Path("experiments/failure_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Failure Analysis — {args.method.upper()} Retrieval\n\n")
        f.write(f"Analysed **{len(sampled)}** failures (Recall@1 = 0) "
                f"out of {len(records)} total evaluated questions.\n\n")
        f.write("## Failure Category Distribution\n\n")
        f.write("| Failure Category | Count | Share |\n")
        f.write("| :--- | :---: | :---: |\n")
        for cat, cnt in counts.most_common():
            pct = 100 * cnt / len(sampled)
            f.write(f"| `{cat}` | {cnt} | {pct:.1f}% |\n")
        f.write(f"| **Total** | **{len(sampled)}** | 100.0% |\n\n")

        f.write("## Category Descriptions\n\n")
        f.write("| Category | Description |\n")
        f.write("| :--- | :--- |\n")
        f.write("| `keyword_mismatch` | Question terms have no lexical overlap with retrieved content — structural retrieval needed |\n")
        f.write("| `wrong_page` | Correct page was retrieved but ranked below position 1 — reranking could fix |\n")
        f.write("| `table_reasoning` | Question requires numerical aggregation or table cell arithmetic |\n")
        f.write("| `visual_ambiguity` | Answer depends on visual features (logo, signature, layout) not captured in text |\n")
        f.write("| `ocr_error` | Ground-truth answer contains unusual character patterns suggesting OCR noise |\n")
        f.write("| `unknown` | Does not fit any above category |\n\n")

        f.write("## Sample Failures\n\n")
        for cat in counts.keys():
            examples = [x for x in categorised if x["category"] == cat][:3]
            f.write(f"### `{cat}`\n\n")
            for ex in examples:
                f.write(f"- **Q**: {ex['question']}\n")
                f.write(f"  **GT page**: {ex['gt_page']} | **GT ans**: {ex['gt_answers'][:2]}\n")
                f.write(f"  **Top retrieved**: `{ex['top_content'][:80]}`\n\n")

    # Save JSON
    json_path = Path("experiments/failure_analysis.json")
    with open(json_path, "w") as f:
        json.dump({
            "method": args.method,
            "total_records": len(records),
            "total_failures": len(failures),
            "analysed": len(sampled),
            "category_counts": dict(counts.most_common()),
            "failures": categorised,
        }, f, indent=2)

    print(f"\n✅  Detailed failures saved to: {json_path}")
    print(f"✅  Markdown summary saved to:  {md_path}")


if __name__ == "__main__":
    main()
