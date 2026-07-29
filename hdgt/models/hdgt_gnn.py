"""
hdgt/models/hdgt_gnn.py

Phase 2 — Heterogeneous Graph Neural Network (HDGT-GNN) for Multimodal Retrieval.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    HeteroData = None


class HDGTHeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for HDGT document graphs.
    Passes message-passing updates across structural, spatial, and sequential edges.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError("torch_geometric is required for HDGTHeteroGNN.")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Linear projections for node types
        self.node_types = ["text", "page", "section", "figure", "table"]
        self.input_projections = nn.ModuleDict({
            nt: nn.Linear(in_dim, hidden_dim) for nt in self.node_types
        })

        # Relational edge types for HeteroConv (including reverse edge types)
        self.edge_types = [
            ("page", "contains", "text"),
            ("text", "contained_in", "page"),
            ("text", "reading_order", "text"),
            ("text", "spatial", "text"),
            ("section", "parent_child", "text"),
            ("text", "child_of", "section"),
        ]

        # Multi-layer HeteroConv layers
        self.convs = nn.ModuleList()
        for l in range(num_layers):
            conv_dict = {}
            for et in self.edge_types:
                conv_dict[et] = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Output representation head
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over HeteroData node features and edge structures.
        
        Parameters
        ----------
        x_dict : Dict[str, Tensor]
            Dictionary mapping node_type -> node feature tensor [N_type, in_dim].
        edge_index_dict : Dict[Tuple, Tensor]
            Dictionary mapping edge_type -> edge index tensor [2, E_type].
            
        Returns
        -------
        Dict[str, Tensor] mapping node_type -> output embedding tensor [N_type, out_dim].
        """
        # Project inputs to hidden dimension
        h_dict = {}
        for nt, x in x_dict.items():
            if nt in self.input_projections:
                h_dict[nt] = F.relu(self.input_projections[nt](x))
            else:
                h_dict[nt] = x

        # Filter edge_index_dict to only include supported edge types present in model
        valid_edge_dict = {
            et: edge_index_dict[et]
            for et in self.edge_types
            if et in edge_index_dict and edge_index_dict[et].numel() > 0
        }

        # Apply message-passing layers
        for conv in self.convs:
            if valid_edge_dict:
                h_dict_next = conv(h_dict, valid_edge_dict)
                for nt in h_dict:
                    if nt in h_dict_next:
                        h_dict[nt] = F.relu(h_dict_next[nt] + h_dict[nt])
                        h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)

        # Output projection
        out_dict = {nt: self.out_proj(h) for nt, h in h_dict.items()}
        return out_dict

    def compute_relevance(
        self,
        query_emb: torch.Tensor,
        node_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity relevance scores between question query and node embeddings.
        
        Parameters
        ----------
        query_emb : Tensor of shape [1, out_dim] or [out_dim]
        node_embs : Tensor of shape [N, out_dim]
        
        Returns
        -------
        Tensor of shape [N] containing similarity scores.
        """
        q_norm = F.normalize(query_emb, p=2, dim=-1)
        n_norm = F.normalize(node_embs, p=2, dim=-1)
        scores = torch.sum(q_norm * n_norm, dim=-1)
        return scores
