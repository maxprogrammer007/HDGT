# HDGT Phase 1.6 — Baseline Comparison for Document Retrieval

**Phase**: 1.6 — MP-DocVQA Dataset Preparation & Retrieval Benchmark Design  
**Status**: Architectural comparison (no quantitative benchmark scores yet — see Phase 5)

---

## Overview

This document surveys retrieval methods that are relevant to the MP-DocVQA benchmark.
For each method we record:

- **Retrieval level**: What unit is returned (page, patch, segment, node)
- **Modalities used**: Text only (T), Vision only (V), or both (T+V)
- **Graph structure**: Whether the method explicitly models document topology
- **Multi-page support**: Whether the method retrieves across multiple pages
- **Evidence links**: Whether the method can link a referring paragraph to the cited figure/table
- **Published ANLS on MP-DocVQA**: Official benchmark scores where reported

---

## 1. Capability Matrix

| Method | Retrieval Level | Modality | Graph | Multi-Page | Evidence Links | MP-DocVQA ANLS |
|--------|----------------|----------|-------|-----------|----------------|----------------|
| **BM25 (text)** | Page | T | ✗ | ✗ (per-page) | ✗ | ~0.42 (est.) |
| **TF-IDF** | Page | T | ✗ | ✗ | ✗ | — |
| **DPR (Dense)** | Page | T | ✗ | ✗ | ✗ | — |
| **ColPali** | Page (patches) | V | ✗ | ✗ | ✗ | 0.601 |
| **VisRAG** | Page | T+V | ✗ | ✗ | ✗ | — |
| **RAG-Anything** | Segment | T+V | Partial | Partial | Partial | — |
| **LayoutLMv3** | Token | T+V | ✗ | ✗ | ✗ | 0.783 (DocVQA) |
| **DocOwl 1.5** | Page | T+V | ✗ | ✓ | ✗ | 0.681 |
| **InternVL3** | Page | T+V | ✗ | ✓ | ✗ | 0.801 |
| **GPT-4o** | Page | T+V | ✗ | ✓ | Implicit | 0.912 |
| **HDGT (Ours)** | **Node (element)** | **T+V** | **✓** | **✓** | **✓** | Phase 5 |

> [!NOTE]
> ANLS scores are sourced from published papers or challenge leaderboards.
> HDGT's ANLS will be measured in Phase 5 on the official val split.
> `—` = not reported on MP-DocVQA specifically.

---

## 2. Method Summaries

### BM25 (Sparse Text Retrieval)
- **How it works**: TF-IDF-style term frequency weighting over page text.
- **Retrieval unit**: Page (concatenated OCR text per page).
- **Limitation**: Purely lexical; no layout awareness, no vision.
- **Role in HDGT evaluation**: Lower-bound text baseline implemented in `BM25Retriever`.

### ColPali
- **How it works**: Late-interaction similarity between query patch embeddings and page-level patch embeddings (PaliGemma backbone).
- **Paper**: Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models", 2024.
- **Retrieval unit**: Page image.
- **Limitation**: No cross-page reasoning; retrieves at page granularity only. Cannot compute ELA.

### VisRAG
- **How it works**: Embed full page images as dense visual vectors; retrieve via maximum inner product search.
- **Retrieval unit**: Page image.
- **Limitation**: No document structure; layout elements are not distinguished.

### RAG-Anything
- **How it works**: Multi-modal document parsing + chunked retrieval. Partial graph for figure-caption pairing.
- **Retrieval unit**: Chunk (text block or image).
- **Limitation**: No typed heterogeneous graph; no l-hop traversal across edge types.

### LayoutLMv3
- **How it works**: Pre-trained on OCR tokens with 2D positional encoding + visual patches.
- **Retrieval unit**: Token sequence (per page).
- **Limitation**: Single-page; multi-page reasoning requires external retrieval.

### DocOwl 1.5
- **How it works**: Multi-page visual document understanding via interleaved page images.
- **Paper**: Hu et al., "mPLUG-DocOwl 1.5", 2024.
- **Limitation**: Implicit attention; no explicit graph for element-level localization.

### InternVL3
- **How it works**: Strong VLM with long-context multi-page document understanding.
- **Limitation**: Token inflation as page count increases; no explicit structural graph.

### GPT-4o (Oracle Ceiling)
- **How it works**: Sends all page images in context; uses implicit vision attention.
- **Limitation**: Closed-source; extremely high cost per inference; not deployable.
- **Role**: Upper-bound ceiling for all open-source methods.

---

## 3. HDGT Architectural Differentiation

| Dimension | Baselines | HDGT |
|-----------|-----------|------|
| Retrieval unit | Page or chunk | **Element node** (paragraph, figure, table) |
| Evidence localization | None (ELA not computable) | **Node content** (ELA measurable) |
| Cross-page linking | Implicit attention | **Explicit reference + continuation edges** |
| Structural hierarchy | None | **parent_child edges** (section → paragraph) |
| Graph type | None or partial | **Typed heterogeneous graph** (5 node types, 6 edge types) |
| Layout awareness | Implicit (attention over tokens) | **Spatial k-NN edges** (explicit 2D proximity) |
| Retrieval interpretability | Black-box | **Traversal path traceable per question** |

---

## 4. Published Baselines on MP-DocVQA

The following table records published results on the **official MP-DocVQA val split**
as of July 2026 (sourced from papers and the RRC leaderboard):

| Method | ANLS (val) | Source |
|--------|:----------:|--------|
| ColPali | 0.601 | Faysse et al., 2024 |
| DocOwl 1.5 (Hires) | 0.681 | Hu et al., 2024 |
| InternVL2-8B | 0.781 | Chen et al., 2024 |
| InternVL3-8B | 0.801 | InternVL3 report, 2025 |
| GPT-4o | 0.912 | OpenAI, 2024 |
| **HDGT (Phase 5)** | **TBD** | This work |

> [!IMPORTANT]
> These scores reflect **end-to-end answer generation**, not retrieval-only metrics.
> HDGT Phase 5 will report Recall@K, MRR, ELA (retrieval), and ANLS (generation)
> separately to isolate the contribution of graph retrieval from the answer generator.

---

## 5. Retrieval-Only Baselines

No publicly available paper reports Recall@K or MRR for MP-DocVQA with element-level
node retrieval. This is a known gap in the literature. HDGT's Phase 5 results will
introduce element-level retrieval metrics (especially ELA) as a novel contribution.

| Metric | BM25 | DPR | ColPali | HDGT-BM25 | HDGT-Graph |
|--------|------|-----|---------|-----------|------------|
| Recall@1 | — | — | — | Phase 5 | Phase 5 |
| Recall@5 | — | — | — | Phase 5 | Phase 5 |
| MRR | — | — | — | Phase 5 | Phase 5 |
| ELA | N/A | N/A | N/A | Phase 5 | Phase 5 |

`N/A` = metric not computable for page-level retrievers.
`—` = not reported in the literature for this dataset.
