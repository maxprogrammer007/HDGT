# Phase 1.6: MP-DocVQA Dataset Preparation & Retrieval Benchmark Design

## Objective

The objective of this phase is to prepare the complete evaluation pipeline for HDGT on the MP-DocVQA benchmark before introducing semantic node representations.

Following the feedback received after Phase 1, this phase focuses on understanding the benchmark, analyzing document structures, defining retrieval units, and preparing fair comparisons with existing retrieval methods.

---

# Motivation

Phase 1 successfully converts heterogeneous PDF documents into multi-relational document graphs.

However, graph construction alone does not demonstrate retrieval capability.

Before integrating Vision-Language Models, we first need to understand

- what the benchmark expects,
- what constitutes a retrieval target,
- how existing methods perform retrieval,
- and how HDGT will be evaluated.

---

# Dataset Overview

Dataset:
MP-DocVQA (Multi-Page Document Visual Question Answering)

Downloaded Components

✓ Questions and Answers

```
qas/
```

✓ Document Images

```
images/
```

✓ OCR Results (Amazon Textract)

```
ocr/
```

✓ Processed Framework Metadata

```
mpdocvqa_imdbs/
```

---

# Dataset Statistics

Version: MP-DocVQA v1.0

- 5,929 documents
- 47,952 pages
- 46,436 question-answer pairs
- OCR extracted using Amazon Textract

The dataset supports multi-page reasoning where the answer may appear on any page within the document.

---

# Goals of Phase 1.6

## Goal 1

Understand the complete MP-DocVQA data format.

Deliverables

- inspect IMDB files
- inspect question annotations
- inspect OCR structure
- inspect page image organization

---

## Goal 2

Map MP-DocVQA documents into the HDGT graph pipeline.

For every document:

PDF (or page images)

↓

Docling parsing

↓

Document Graph

↓

Question alignment

↓

Evidence Nodes

---

## Goal 3

Identify retrieval granularity.

Investigate whether retrieval should operate at

- document level
- page level
- section level
- paragraph level
- caption level
- figure level

This analysis directly addresses the reviewer's question regarding semantic relationships among document elements.

---

## Goal 4

Study existing retrieval baselines.

Collect results from literature for

- MP-DocVQA
- DocVQA
- InfographicVQA (if applicable)

Methods include

- ColPali
- VisRAG
- RAG-Anything
- MinerU2.5
- DocOwl
- InternVL
- GPT-4o based retrieval
- LayoutLM family
- DocFormer

Record

- ANLS
- Recall@K
- MRR
- Retrieval strategy

---

## Goal 5

Design HDGT retrieval protocol.

The protocol will define

Input

Question

↓

Semantic retrieval

↓

Graph traversal

↓

Evidence subgraph

↓

Answer generation

Evaluation Metrics

- ANLS
- Recall@K
- MRR
- Evidence Localization Accuracy (ELA)
- Graph Traversal Length

---

## Goal 6

Prepare evaluation scripts.

Develop scripts for

- loading MP-DocVQA
- loading HDGT graphs
- mapping questions to documents
- computing retrieval metrics
- visualizing retrieved evidence

No Vision-Language Models are required in this phase.

---

# Expected Deliverables

✓ Dataset exploration notebook

✓ MP-DocVQA parser

✓ Question loader

✓ OCR loader

✓ Graph loader

✓ Retrieval evaluation pipeline

✓ Baseline comparison table

✓ Retrieval protocol

✓ Dataset statistics report

---

# Success Criteria

Phase 1.6 is complete when

- MP-DocVQA is fully understood
- Every document can be mapped into an HDGT graph
- Questions can be linked to graph nodes
- Evaluation scripts are ready
- Baseline methods are documented
- Retrieval protocol is finalized

At this point, the project will be ready for semantic encoding using Vision-Language Models.

---

# Phase 2 Preview

Once Phase 1.6 is complete, Phase 2 will begin.

Phase 2 introduces

- Qwen2.5-VL
- Vision-language node embeddings
- Multimodal feature fusion
- Graph Neural Network encoding
- Semantic retrieval over HDGT graphs

The graph construction and evaluation pipeline developed in Phases 1, 1.5, and 1.6 will remain unchanged.