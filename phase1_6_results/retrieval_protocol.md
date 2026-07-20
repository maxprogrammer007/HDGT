# HDGT Phase 1.6 — Retrieval Evaluation Protocol (Finalized)

**Phase**: 1.6 — MP-DocVQA Dataset Preparation & Retrieval Benchmark Design  
**Supersedes**: `phase1_5_results/retrieval_protocol.md` (Phase 1.5 draft)  
**Status**: Finalized — scripts implemented and ready for workstation execution

---

## 1. Dataset

### Primary Benchmark: MP-DocVQA

| Property | Value |
|----------|-------|
| Version | v1.0 (Feb 2023) |
| Total QA pairs | 46,436 |
| Unique documents | 5,929 |
| Unique page-range contexts | 8,989 |
| Pages per context (avg) | 8.41 |
| Pages per context (max) | 20 |
| Evaluation split | `val` (5,187 questions) |
| Test split | Withheld (submit to RRC portal for ANLS) |

See `phase1_6_results/dataset_statistics.md` for the full breakdown.

---

## 2. Retrieval Task Definition

Given:
- A natural language question **q**
- A pre-built HDGT graph **G_D = (V_D, E_D)** for document **D**

Retrieve:
> A ranked list of document element nodes **[n₁, n₂, ..., nₖ]** ⊆ V_D
> such that the nodes most likely to contain the answer appear first.

Each returned node has:
- `node_uid` — globally unique string ID
- `type` — text | table | figure | section | page
- `page` — 0-indexed page number within the context
- `content` — raw text string (paragraph, table cell text, caption)
- `bbox` — [x1, y1, x2, y2] normalised to [0, 1]
- `score` — retrieval confidence

---

## 3. Evaluation Metrics

### 3.1 Recall@K
$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[p^* \in \text{pages}(\hat{E}_{1:K})]$$

- **K values**: 1, 5, 10
- **Ground truth**: `answer_page_idx` (0-indexed position in the context page list)
- **Mapping**: Retrieved node's `page` attribute is compared to `answer_page_idx`

### 3.2 Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(p^*)}$$

Where rank(p*) is the position of the first retrieved node on the correct page.

### 3.3 Evidence Localization Accuracy (ELA) — Proposed
$$\text{ELA} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[\text{answer} \in \hat{n}_1.\text{content}]$$

- Top-1 retrieved node's `content` is checked for substring match (normalised).
- **Why novel**: Page-level retrievers cannot compute ELA because they return full pages, not element text.
- ELA is an HDGT-specific contribution being introduced in Phase 1.6.

### 3.4 ANLS (Answer Quality — Phase 4+)
$$\text{ANLS} = \frac{1}{|Q|} \sum_{q \in Q} \max_{a^* \in A^*} \left(1 - \frac{\text{EditDist}(\hat{a}, a^*)}{\max(|\hat{a}|, |a^*|)}\right)$$

- Requires a trained generator (Phase 4).
- In Phase 1.6, ANLS is computed using the top retrieved node's `content` as a proxy answer.

---

## 4. Retrieval Methods

### 4.1 Methods Implemented in Phase 1.6

| Method | Class | Description |
|--------|-------|-------------|
| Random | `RandomRetriever` | Uniform random node scores — lower bound |
| BM25 | `BM25Retriever` | Sparse text retrieval over node contents |
| Mock-Graph | `MockGraphRetriever` | Character-overlap anchor + l-hop BFS traversal |

### 4.2 Phase 5 Methods (require Phase 2 embeddings)

| Method | Description |
|--------|-------------|
| Dense node retrieval | Query vs. Qwen2.5-VL node embeddings (cosine sim) |
| HDGT-Full | Dense anchor + graph traversal with typed edge priorities |
| ColPali-style | Page-level vision retrieval (for comparison) |
| DPR-style | Dense page-level text retrieval (for comparison) |

---

## 5. Ablation Study Design

| Experiment | Change from HDGT-Full | Expected Effect |
|------------|----------------------|-----------------|
| HDGT-NoGraph | Remove BFS traversal | Lower Recall@K for multi-hop Qs |
| HDGT-NoRef | Remove `reference` edges | Lower ELA for figure/table Qs |
| HDGT-NoParent | Remove `parent_child` edges | Lower performance on section Qs |
| HDGT-NoCont | Remove `continuation` edges | Lower performance on multi-page tables |
| BM25 | Text-only sparse retrieval | Baseline comparison |
| Random | Random node selection | Lower bound |

---

## 6. Execution Commands

### Step 1: Dataset preparation (workstation)
```bash
python prepare_mpdocvqa.py \
  --data-root /home/cvpruts/Downloads/HDGT-main\(1\)/HDGT-main/data/MP-DocVQA
```

### Step 2: Graph construction (workstation)
```bash
# Full val split
python build_mpdocvqa_graphs.py \
  --data-root /home/cvpruts/Downloads/HDGT-main\(1\)/HDGT-main/data/MP-DocVQA \
  --split val

# Smoke test (first 50 contexts)
python build_mpdocvqa_graphs.py \
  --data-root /home/cvpruts/Downloads/HDGT-main\(1\)/HDGT-main/data/MP-DocVQA \
  --limit 50
```

### Step 3: Run evaluation
```bash
# BM25 on val (first 100 questions, quick test)
python evaluate_retrieval.py \
  --data-root /home/cvpruts/Downloads/HDGT-main\(1\)/HDGT-main/data/MP-DocVQA \
  --split val --method bm25 --limit 100

# Mock graph traversal
python evaluate_retrieval.py \
  --data-root /home/cvpruts/Downloads/HDGT-main\(1\)/HDGT-main/data/MP-DocVQA \
  --split val --method mock-graph --limit 100

# Random baseline
python evaluate_retrieval.py \
  --data-root /home/cvpruts/Downloads/HDGT-main\(1\)/HDGT-main/data/MP-DocVQA \
  --split val --method random
```

### Step 4: Run unit tests (local — no dataset needed)
```bash
python -m pytest tests/test_evaluation_pipeline.py -v
```

### Step 5: Generate dataset statistics (local — only needs qas.zip)
```bash
python generate_phase16_stats.py --data-root data/MP-DocVQA
```

---

## 7. Output Format

Each call to `evaluate_retrieval.py` saves a JSONL file at:
```
experiments/retrieval_results_{split}_{method}.jsonl
```

Each line is one question result:
```json
{
  "question_id": "49153",
  "query": "What is the 'actual' value per 1000, during the year 1975?",
  "context_id": "pybv0228_p80_p80",
  "ground_truth_page": 0,
  "ground_truth_answers": ["0.28"],
  "retrieved_nodes": [
    {
      "node_uid": "pybv0228_p80_p80_p0_n3",
      "type": "table",
      "page": 0,
      "rank": 1,
      "score": 4.712,
      "content": "Year | Actual | Budget\n1975 | 0.28 | 0.31",
      "bbox": [0.1, 0.3, 0.9, 0.7]
    }
  ],
  "metrics": {
    "recall@1": 1.0,
    "recall@5": 1.0,
    "recall@10": 1.0,
    "mrr": 1.0,
    "ela": 1.0,
    "anls": 0.82
  }
}
```

---

## 8. Phase 1.6 → Phase 2 Handoff

Phase 1.6 is complete when:

- [x] `qas.zip` fully parsed — all 46,436 questions mapped
- [x] `dataset_statistics.md` compiled
- [x] `baseline_comparison.md` with published ANLS scores
- [x] `loaders.py`, `metrics.py`, `retriever.py` implemented
- [x] `evaluate_retrieval.py` CLI ready
- [x] Unit tests passing (`tests/test_evaluation_pipeline.py`)
- [ ] Graphs compiled on workstation (`build_mpdocvqa_graphs.py`)
- [ ] BM25 + Random baseline results recorded on `val` split
- [ ] Mock-graph traversal results recorded on `val` split

**Phase 2 begins** once BM25 and random baseline numbers are in hand.
Phase 2 replaces the character-overlap proxy scores in `MockGraphRetriever`
with real Qwen2.5-VL node embeddings, enabling dense retrieval.
