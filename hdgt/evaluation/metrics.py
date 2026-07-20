"""
hdgt/evaluation/metrics.py

Phase 1.6 — Retrieval evaluation metrics for HDGT.

All functions are pure Python with no PyTorch / model dependencies.

Metrics
-------
recall_at_k     : Fraction of questions where the correct page appears in top-K results
mrr             : Mean Reciprocal Rank of the first correct page hit
ela             : Evidence Localization Accuracy (answer substring in top-node content)
anls            : Average Normalized Levenshtein Similarity (for generation quality)
compute_metrics : Convenience wrapper that computes all metrics at once
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Levenshtein distance (no external dependency)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _normalise_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, and unicode-normalise."""
    text = unicodedata.normalize("NFKD", text).lower().strip()
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_pages: List[int],
    ground_truth_page: int,
    k: int = 5,
) -> float:
    """
    1.0 if `ground_truth_page` appears in the first `k` retrieved page numbers,
    else 0.0.

    Parameters
    ----------
    retrieved_pages : List[int]
        Ordered list of page numbers returned by a retriever (0-indexed).
    ground_truth_page : int
        The correct answer page (0-indexed, as in MP-DocVQA answer_page_idx).
    k : int
        Cutoff rank. Evaluated at k=1, 5, 10 in the benchmark protocol.

    Returns
    -------
    float : 1.0 or 0.0
    """
    if not retrieved_pages:
        return 0.0
    return float(ground_truth_page in retrieved_pages[:k])


# ---------------------------------------------------------------------------
# MRR — Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def mrr(
    retrieved_pages: List[int],
    ground_truth_page: int,
) -> float:
    """
    Reciprocal rank of the first hit for `ground_truth_page` in the list.

    Returns 0.0 if not found.
    """
    for rank, page in enumerate(retrieved_pages, start=1):
        if page == ground_truth_page:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# ELA — Evidence Localization Accuracy
# ---------------------------------------------------------------------------

def ela(
    retrieved_contents: List[str],
    ground_truth_answers: List[str],
) -> float:
    """
    Evidence Localization Accuracy.

    Returns 1.0 if ANY ground-truth answer string is a substring of the top
    retrieved node's content (case-insensitive, normalised). Otherwise 0.0.

    This is an element-level metric unique to HDGT — page-level retrievers
    cannot compute it because they do not return individual node text.

    Parameters
    ----------
    retrieved_contents : List[str]
        Ordered content strings from the retriever (top node first).
    ground_truth_answers : List[str]
        All valid answer strings for this question.

    Returns
    -------
    float : 1.0 or 0.0
    """
    if not retrieved_contents or not ground_truth_answers:
        return 0.0
    top_content = _normalise_text(retrieved_contents[0])
    for ans in ground_truth_answers:
        if _normalise_text(ans) in top_content:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# ANLS — Average Normalised Levenshtein Similarity
# ---------------------------------------------------------------------------

def anls(
    predicted_text: str,
    ground_truth_answers: List[str],
    threshold: float = 0.5,
) -> float:
    """
    Compute ANLS between `predicted_text` and the best ground-truth answer.

    ANLS = max over all GT answers of:
        1 - EditDist(pred, gt) / max(len(pred), len(gt))
        if the ratio ≥ threshold, else 0.

    Reference: Biten et al., "Scene Text Visual Question Answering", ICCV 2019.

    Parameters
    ----------
    predicted_text : str
        The system's predicted answer string.
    ground_truth_answers : List[str]
        All valid ground-truth answer strings.
    threshold : float
        Similarity below this value is set to 0 (penalty for hallucinated answers).

    Returns
    -------
    float in [0.0, 1.0]

    Notes
    -----
    In Phase 1.6, the generator module (Phase 4) does not yet exist.
    Pass the content of the top retrieved node as `predicted_text` to test
    the code path; ANLS will be low but the metric computation will be correct.
    """
    if not predicted_text or not ground_truth_answers:
        return 0.0

    pred_norm = _normalise_text(predicted_text)
    best = 0.0
    for gt in ground_truth_answers:
        gt_norm = _normalise_text(gt)
        max_len = max(len(pred_norm), len(gt_norm))
        if max_len == 0:
            score = 1.0
        else:
            dist = _edit_distance(pred_norm, gt_norm)
            nls = 1.0 - dist / max_len
            score = nls if nls >= threshold else 0.0
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_metrics(
    retrieved_pages: List[int],
    retrieved_contents: List[str],
    ground_truth_page: int,
    ground_truth_answers: List[str],
    predicted_text: Optional[str] = None,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, Any]:
    """
    Compute all Phase 1.6 metrics in one call.

    Parameters
    ----------
    retrieved_pages    : page numbers in ranked order (top first)
    retrieved_contents : content strings in ranked order (top first)
    ground_truth_page  : correct answer page (0-indexed)
    ground_truth_answers : list of valid answer strings
    predicted_text     : optional generated answer (for ANLS)
    k_values           : list of K values for Recall@K

    Returns
    -------
    dict with keys: recall@1, recall@5, recall@10, mrr, ela, anls
    """
    results: Dict[str, Any] = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_pages, ground_truth_page, k)
    results["mrr"]  = mrr(retrieved_pages, ground_truth_page)
    results["ela"]  = ela(retrieved_contents, ground_truth_answers)
    results["anls"] = anls(
        predicted_text or (retrieved_contents[0] if retrieved_contents else ""),
        ground_truth_answers,
    )
    return results
