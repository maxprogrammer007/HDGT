"""
compute_confidence_intervals.py

HDGT Priority 9 — Statistical Confidence Intervals

Bootstrap confidence intervals (95%) over evaluation JSONL files.
Reports: Mean ± 95% CI for Recall@1, Recall@5, Recall@10, MRR, ELA.

Usage:
    python compute_confidence_intervals.py --methods bm25
    python compute_confidence_intervals.py --methods bm25 phase2_hdgt
    python compute_confidence_intervals.py --n_bootstrap 2000
"""

import argparse
import json
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for HDGT evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--methods", nargs="+", default=["bm25"],
                   help="Method name(s) matching experiments/retrieval_results_val_{method}.jsonl")
    p.add_argument("--n_bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples.")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level (default 0.95 = 95%% CI).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, confidence: float,
                 seed: int = 42) -> tuple[float, float, float]:
    """
    Compute mean and bootstrap CI for `values`.
    Returns (mean, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lower), float(upper)


def load_metrics(jsonl_path: Path, metric_keys: list) -> dict[str, np.ndarray]:
    data = {k: [] for k in metric_keys}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            m = r.get("metrics", {})
            for k in metric_keys:
                data[k].append(m.get(k, 0.0))
    return {k: np.array(v) for k, v in data.items()}


def print_ci_table(method: str, cis: dict, n: int, confidence: float):
    pct = int(confidence * 100)
    print(f"\n  Method: {method}   (n={n:,}, {pct}% CI, 1000 bootstrap samples)")
    print(f"  {'Metric':<14} {'Mean':>8} {'Lower':>8} {'Upper':>8}  {'95% CI Width':>12}")
    print("  " + "-" * 58)
    for k, (mean, lo, hi) in cis.items():
        scale = 100.0 if "recall" in k else 1.0
        unit  = "%" if "recall" in k else ""
        width = (hi - lo) * scale
        print(f"  {k.upper():<14} "
              f"{mean*scale:>7.2f}{unit}  "
              f"{lo*scale:>7.2f}{unit}  "
              f"{hi*scale:>7.2f}{unit}  "
              f"  ±{width/2:.2f}{unit}")


def main():
    args = parse_args()
    metric_keys = ["recall@1", "recall@5", "recall@10", "mrr", "ela"]
    out_dir = Path("experiments")

    all_results = {}

    print(f"\n{'='*65}")
    print("  HDGT Evaluation — Bootstrap Confidence Intervals")
    print(f"{'='*65}")

    for method in args.methods:
        jsonl_path = out_dir / f"retrieval_results_val_{method}.jsonl"
        if not jsonl_path.exists():
            print(f"\n  ⚠️  File not found: {jsonl_path}  (skipping)")
            continue

        metrics_data = load_metrics(jsonl_path, metric_keys)
        n = len(next(iter(metrics_data.values())))

        cis = {}
        for k, vals in metrics_data.items():
            mean, lo, hi = bootstrap_ci(vals, args.n_bootstrap,
                                        args.confidence, args.seed)
            cis[k] = (mean, lo, hi)

        print_ci_table(method, cis, n, args.confidence)
        all_results[method] = {"n": n, "confidence": args.confidence, "cis": {
            k: {"mean": v[0], "lower": v[1], "upper": v[2]}
            for k, v in cis.items()
        }}

    # Side-by-side comparison if multiple methods
    if len(args.methods) >= 2 and len(all_results) >= 2:
        print(f"\n\n  SIDE-BY-SIDE COMPARISON (95% CI)")
        print(f"  {'Metric':<14}", end="")
        for m in args.methods:
            if m in all_results:
                print(f"  {m[:18]:<22}", end="")
        print()
        print("  " + "-" * (14 + 24 * len(all_results)))
        for k in metric_keys:
            scale = 100.0 if "recall" in k else 1.0
            unit  = "%" if "recall" in k else ""
            print(f"  {k.upper():<14}", end="")
            for m in args.methods:
                if m not in all_results:
                    continue
                ci = all_results[m]["cis"][k]
                mean, lo, hi = ci["mean"], ci["lower"], ci["upper"]
                print(f"  {mean*scale:.2f}{unit} [{lo*scale:.2f}–{hi*scale:.2f}]{unit:1}   ", end="")
            print()

    # Save JSON
    out_path = out_dir / "confidence_intervals.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save markdown
    md_path = out_dir / "confidence_intervals.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Results with Bootstrap Confidence Intervals\n\n")
        f.write(f"> {int(args.confidence*100)}% confidence intervals, {args.n_bootstrap} bootstrap resamples.\n\n")
        for method, data in all_results.items():
            f.write(f"## {method} (n = {data['n']:,})\n\n")
            f.write("| Metric | Mean | 95% Lower | 95% Upper | ± Width |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for k, ci in data["cis"].items():
                scale = 100.0 if "recall" in k else 1.0
                unit  = "%" if "recall" in k else ""
                width = (ci["upper"] - ci["lower"]) * scale / 2
                f.write(f"| {k.upper()} | "
                        f"{ci['mean']*scale:.2f}{unit} | "
                        f"{ci['lower']*scale:.2f}{unit} | "
                        f"{ci['upper']*scale:.2f}{unit} | "
                        f"±{width:.2f}{unit} |\n")
            f.write("\n")

    print(f"\n✅  Results saved to:")
    print(f"   {out_path}")
    print(f"   {md_path}")


if __name__ == "__main__":
    main()
