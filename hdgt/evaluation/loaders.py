"""
hdgt/evaluation/loaders.py

Phase 1.6 — Data loaders for MP-DocVQA and HDGT graph files.

Classes
-------
MPDocVQALoader
    Reads the QA annotation ZIP directly (no manual extraction needed).
    Yields dicts with question, answers, doc_id, page_ids, answer_page_idx.

GraphLoader
    Loads a PyG HeteroData object from experiments/mpdocvqa/<context_id>_graph.pt.
    Returns None if the graph file has not been compiled yet.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Generator, List, Optional

try:
    import torch
    from torch_geometric.data import HeteroData
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False
    HeteroData = None  # type: ignore


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MPDocVQALoader
# ---------------------------------------------------------------------------

class MPDocVQALoader:
    """
    Lazily iterate over MP-DocVQA QA pairs from the qas.zip archive.

    Parameters
    ----------
    data_root : str | Path
        Path to the MP-DocVQA root directory (must contain qas.zip).
    split : str
        One of 'train', 'val', 'test'.
    limit : int | None
        If set, yield at most `limit` questions. Useful for fast dev runs.

    Usage
    -----
    >>> loader = MPDocVQALoader("data/MP-DocVQA", split="val", limit=50)
    >>> for item in loader:
    ...     print(item["question"], item["answers"])
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        data_root: str | Path = "data/MP-DocVQA",
        split: str = "val",
        limit: Optional[int] = None,
    ) -> None:
        self.data_root = Path(data_root)
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got {split!r}")
        self.split = split
        self.limit = limit
        self._data: Optional[List[dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[dict, None, None]:
        """Yield one QA item at a time."""
        data = self._load()
        count = 0
        for item in data:
            if self.limit is not None and count >= self.limit:
                break
            yield self._normalise(item)
            count += 1

    def __len__(self) -> int:
        data = self._load()
        if self.limit is not None:
            return min(self.limit, len(data))
        return len(data)

    def build_context_id(self, page_ids: List[str]) -> str:
        """
        Derive the context_id key used as the graph filename stem.

        The context ID format mirrors prepare_mpdocvqa.py:
            {doc_id}_p{start_page}_p{end_page}
        """
        doc_id = page_ids[0].rsplit("_p", 1)[0]
        pnums = []
        for pid in page_ids:
            match = re.search(r"_p(\d+)$", pid)
            pnums.append(int(match.group(1)) if match else 0)
        start_p = min(pnums)
        end_p = max(pnums)
        return f"{doc_id}_p{start_p}_p{end_p}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> List[dict]:
        """Load (and cache) QA data from qas.zip."""
        if self._data is not None:
            return self._data

        zip_path = self.data_root / "qas.zip"
        json_name = f"{self.split}.json"

        # Try extracted file first (faster)
        extracted = self.data_root / json_name
        if extracted.exists():
            with open(extracted, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._data = raw["data"]
            logger.info(f"Loaded {len(self._data)} items from {extracted}")
            return self._data

        # Fall back to reading from the ZIP
        if not zip_path.exists():
            raise FileNotFoundError(
                f"qas.zip not found at {zip_path}. "
                "Download from https://rrc.cvc.uab.es/?ch=17"
            )
        with zipfile.ZipFile(zip_path, "r") as zf:
            if json_name not in zf.namelist():
                raise KeyError(
                    f"{json_name} not found inside {zip_path}. "
                    f"Available: {zf.namelist()}"
                )
            with zf.open(json_name) as f:
                raw = json.load(f)

        self._data = raw["data"]
        logger.info(f"Loaded {len(self._data)} items from {zip_path}::{json_name}")
        return self._data

    @staticmethod
    def _normalise(item: dict) -> dict:
        """Standardise key names and ensure required fields exist."""
        return {
            "question_id":    str(item.get("questionId", "")),
            "question":       item.get("question", ""),
            "doc_id":         item.get("doc_id", ""),
            "page_ids":       item.get("page_ids", []),
            "answers":        item.get("answers", []),
            "answer_page_idx": item.get("answer_page_idx", None),
            "data_split":     item.get("data_split", ""),
        }


# ---------------------------------------------------------------------------
# GraphLoader
# ---------------------------------------------------------------------------

class GraphLoader:
    """
    Load a pre-built HDGT HeteroData graph from the experiments/mpdocvqa/ directory.

    Parameters
    ----------
    graphs_dir : str | Path
        Directory containing {context_id}_graph.pt files.

    Usage
    -----
    >>> loader = GraphLoader("experiments/mpdocvqa")
    >>> graph = loader.load("pybv0228_p80_p80")
    >>> if graph is not None:
    ...     print(graph.node_types)
    """

    def __init__(self, graphs_dir: str | Path = "experiments/mpdocvqa") -> None:
        self.graphs_dir = Path(graphs_dir)

    def load(self, context_id: str) -> Optional[HeteroData]:
        """
        Load and return the HeteroData graph for the given context_id.
        Returns None if the file does not exist (graph not yet compiled).
        """
        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch and torch_geometric are required to load graphs. "
                "Install them on the workstation with: "
                "pip install torch torch-geometric"
            )
        path = self.graphs_dir / f"{context_id}_graph.pt"
        if not path.exists():
            # Fallback: match by doc_id prefix
            doc_id = context_id.split("_p")[0]
            matches = list(self.graphs_dir.glob(f"{doc_id}_*_graph.pt"))
            if matches:
                path = matches[0]
            else:
                logger.debug(f"Graph not found: {path}")
                return None
        try:
            data = torch.load(path, weights_only=False)
            return data
        except Exception as e:
            logger.warning(f"Failed to load graph {path}: {e}")
            return None

    def available_ids(self) -> List[str]:
        """Return list of context IDs for which graph files exist."""
        stems = [p.stem for p in self.graphs_dir.glob("*_graph.pt")]
        # strip the trailing '_graph' suffix
        return [s[:-6] if s.endswith("_graph") else s for s in stems]

    def count(self) -> int:
        """Number of compiled graphs in the directory."""
        return sum(1 for _ in self.graphs_dir.glob("*_graph.pt"))


# ---------------------------------------------------------------------------
# Convenience: collect all nodes from a HeteroData into a flat list of dicts
# ---------------------------------------------------------------------------

def graph_to_node_list(graph: HeteroData) -> List[Dict]:
    """
    Flatten all nodes from a HeteroData graph into a list of dicts.

    Each dict contains:
      node_uid, type, role, page, bbox, content

    Useful for BM25 indexing and ELA checks.

    Note: PyG NodeStorage exposes tensor attributes via __getattr__, but
    list-valued attributes (contents, pages, node_uids, etc.) are stored
    in the internal _mapping dict and must be accessed via _mapping.get().
    """
    nodes = []
    for ntype in graph.node_types:
        ndata = graph[ntype]
        n = ndata.x.shape[0]

        # Use _mapping.get() to correctly retrieve list-valued attributes
        mapping = ndata._mapping
        contents = mapping.get("contents", [""] * n)
        bboxes   = mapping.get("bboxes",   [[0, 0, 0, 0]] * n)
        pages    = mapping.get("pages",    [0] * n)
        uids     = mapping.get("node_uids", [""] * n)
        roles    = mapping.get("roles",    [""] * n)

        for i in range(n):
            nodes.append({
                "node_uid": uids[i],
                "type":     ntype,
                "role":     roles[i],
                "page":     pages[i],
                "bbox":     bboxes[i],
                "content":  contents[i],
                "local_idx": i,
            })
    return nodes
