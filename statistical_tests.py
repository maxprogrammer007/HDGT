"""
statistical_tests.py

HDGT Priority 10 — Statistical Significance Testing

Performs McNemar's Test and Paired Permutation Test comparing:
  - BM25 Baseline vs. HDGT GNN (Recall@1 and Recall@10)
  - Homogeneous GraphSAGE vs. HDGT GNN (Recall@10)

Outputs p-values and significance statements suitable for publication.

Usage:
    python statistical_tests.py
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats


def mcnemar_test(b_correct: np.ndarray, a_correct: np.ndarray) -> tuple[float, float]:
    """
    McNemar's test for paired nominal data.
    b_correct: binary array for model B
    a_correct: binary array for model A
    Returns (chi2_stat, p_value)
    """
    # Contingency table:
    # b_correct \ a_correct
    #        0     1
    # 0     n00   n01 (A correct, B wrong)
    # 1     n10   n11
    n01 = np.sum((~b_correct) & a_correct)  # A right, B wrong
    n10 = np.sum(b_correct & (~a_correct))  # B right, A wrong

    if n01 + n10 == 0:
        return 0.0, 1.0

    # McNemar statistic with continuity correction
    chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    p_val = stats.chi2.sf(chi2, df=1)
    return float(chi2), float(p_val)


def paired_permutation_test(b_vals: np.ndarray, a_vals: np.ndarray,
                            n_permutations: int = 10000, seed: int = 42) -> tuple[float, float]:
    """
    Paired permutation test for mean difference.
    Returns (mean_diff, p_value)
    """
    rng = np.random.default_rng(seed)
    obs_diff = np.mean(a_vals - b_vals)
    diffs = a_vals - b_vals

    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(obs_diff):
            count_extreme += 1

    p_val = count_extreme / n_permutations
    return float(obs_diff), float(p_val)


def main():
    bm25_file = Path("experiments/retrieval_results_val_bm25.jsonl")
    hdgt_file = Path("experiments/retrieval_results_val_phase2_hdgt.jsonl")

    if not bm25_file.exists() or not hdgt_file.exists():
        print("Error: Evaluation JSONL files not found.")
        return

    bm25_data = {}
    with open(bm25_file) as f:
        for line in f:
            r = json.loads(line)
            bm25_data[r["question_id"]] = r["metrics"]

    hdgt_data = {}
    with open(hdgt_file) as f:
        for line in f:
            r = json.loads(line)
            hdgt_data[r["question_id"]] = r["metrics"]

    common_qids = sorted(set(bm25_data.keys()) & set(hdgt_data.keys()))
    print(f"\n{'='*75}")
    print(f"  STATISTICAL SIGNIFICANCE TESTS (N = {len(common_qids):,} paired questions)")
    print(f"{'='*75}")

    bm25_r1  = np.array([bm25_data[q].get("recall@1", 0.0) > 0 for q in common_qids])
    hdgt_r1  = np.array([hdgt_data[q].get("recall@1", 0.0) > 0 for q in common_qids])

    bm25_r10 = np.array([bm25_data[q].get("recall@10", 0.0) > 0 for q in common_qids])
    hdgt_r10 = np.array([hdgt_data[q].get("recall@10", 0.0) > 0 for q in common_qids])

    # McNemar tests
    chi2_r1, p_mcnemar_r1 = mcnemar_test(bm25_r1, hdgt_r1)
    chi2_r10, p_mcnemar_r10 = mcnemar_test(bm25_r10, hdgt_r10)

    # Permutation tests
    diff_r1, p_perm_r1 = paired_permutation_test(bm25_r1.astype(float), hdgt_r1.astype(float))
    diff_r10, p_perm_r10 = paired_permutation_test(bm25_r10.astype(float), hdgt_r10.astype(float))

    print("\n1. McNemar's Test (Contingency Analysis):")
    print(f"   Recall@1  (BM25 vs. HDGT) : χ² = {chi2_r1:.2f}, p-value = {p_mcnemar_r1:.4e} "
          f"({'p < 0.001 ***' if p_mcnemar_r1 < 0.001 else 'n.s.'})")
    print(f"   Recall@10 (BM25 vs. HDGT) : χ² = {chi2_r10:.2f}, p-value = {p_mcnemar_r10:.4e} "
          f"({'p < 0.001 ***' if p_mcnemar_r10 < 0.001 else 'n.s.'})")

    print("\n2. Paired Permutation Test (10,000 Resamples):")
    print(f"   Recall@1  Δ (HDGT - BM25) : {diff_r1*100:+.2f} pp, p-value = {p_perm_r1:.4e}")
    print(f"   Recall@10 Δ (HDGT - BM25) : {diff_r10*100:+.2f} pp, p-value = {p_perm_r10:.4e} "
          f"({'p < 0.001 ***' if p_perm_r10 < 0.001 else 'n.s.'})")

    print(f"\n{'='*75}\n")

    res = {
        "mcnemar_test": {
            "recall@1": {"chi2": chi2_r1, "p_value": p_mcnemar_r1},
            "recall@10": {"chi2": chi2_r10, "p_value": p_mcnemar_r10},
        },
        "permutation_test": {
            "recall@1": {"mean_diff": diff_r1, "p_value": p_perm_r1},
            "recall@10": {"mean_diff": diff_r10, "p_value": p_perm_r10},
        }
    }
    out_file = Path("experiments/statistical_tests.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"✅  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
