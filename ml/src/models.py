"""
ml/src/models.py
Core neural architectures for the circuit-supergate compatibility model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class AIGNodeEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.node_type_embedding = nn.Embedding(3, embedding_dim)
        self.inverted_pred_embedding = nn.Embedding(3, embedding_dim)
        nn.init.xavier_uniform_(self.node_type_embedding.weight.data)
        nn.init.xavier_uniform_(self.inverted_pred_embedding.weight.data)

    def forward(self, node_type: torch.Tensor, num_inverted: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.node_type_embedding(node_type),
            self.inverted_pred_embedding(num_inverted)
        ], dim=1)


class AIGGraphEncoder(nn.Module):
    def __init__(self, 
                 node_embedding_dim: int = 32,
                 num_layers: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 128,
                 dropout_rate: float = 0.1):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError("torch_geometric is required for AIGGraphEncoder.")
            
        self.num_layers = num_layers
        self.node_encoder = AIGNodeEncoder(embedding_dim=node_embedding_dim)
        input_dim = node_embedding_dim * 2
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(1, num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.norms.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = output_dim * 2

    def forward(self, batched_data) -> torch.Tensor:
        h = self.node_encoder(
            batched_data.node_type,
            batched_data.num_inverted_predecessors
        )
        edge_index = batched_data.edge_index
        batch = batched_data.batch
        
        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index)
            h = self.norms[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
                
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        return torch.cat([h_max, h_sum], dim=1)

    def get_output_dim(self) -> int:
        return self.output_dim


class GNNCompatibilityModel(nn.Module):
    def __init__(self,
                 node_embedding_dim: int = 32,
                 gnn_layers: int = 3,
                 gnn_hidden_dim: int = 128,
                 circuit_embedding_dim: int = 128,
                 fusion_hidden_dim: int = 256,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.circuit_encoder = AIGGraphEncoder(
            node_embedding_dim=node_embedding_dim,
            num_layers=gnn_layers,
            hidden_dim=gnn_hidden_dim,
            output_dim=circuit_embedding_dim,
            dropout_rate=dropout_rate
        )
        circuit_emb_dim = self.circuit_encoder.get_output_dim()
        
        # 32-bit truth table encoder MLP
        self.truth_table_encoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 3 scalar metrics: normalized area, max delay, pin count
        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        combined_dim = circuit_emb_dim + 64 + 32  # 256 + 96 = 352
        
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode_circuit(self, circuit_graph_batch) -> torch.Tensor:
        return self.circuit_encoder(circuit_graph_batch)

    def score_from_embedding(self, circuit_emb: torch.Tensor, supergate_data: dict) -> torch.Tensor:
        tt_emb = self.truth_table_encoder(supergate_data['truth_table'])
        scalar_features = torch.stack([
            supergate_data['area'],
            supergate_data['delay'],
            supergate_data['num_inputs'].float(),
        ], dim=1)
        scalar_emb = self.scalar_encoder(scalar_features)
        supergate_emb = torch.cat([tt_emb, scalar_emb], dim=1)
        
        if circuit_emb.size(0) == 1 and supergate_emb.size(0) > 1:
            circuit_emb = circuit_emb.expand(supergate_emb.size(0), -1)
            
        combined = torch.cat([circuit_emb, supergate_emb], dim=1)
        return self.fusion_network(combined).squeeze(1)

    def forward(self, circuit_graph_batch, supergate_data: dict) -> torch.Tensor:
        circuit_emb = self.encode_circuit(circuit_graph_batch)
        return self.score_from_embedding(circuit_emb, supergate_data)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
        bce = -targets * torch.log(inputs) - (1.0 - targets) * torch.log(1.0 - inputs)
        p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss