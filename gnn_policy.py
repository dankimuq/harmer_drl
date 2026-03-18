"""
gnn_policy.py

PyTorch-only GNN-style actor-critic skeleton for penetration-testing graphs.
This avoids extra dependencies such as PyTorch Geometric while providing a
clean starting point for graph-based policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh_features = adjacency @ node_features / degree
        return self.self_proj(node_features) + self.neigh_proj(neigh_features)


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, layers: int = 3):
        super().__init__()
        convs = [SimpleGraphConv(node_dim, hidden_dim)]
        for _ in range(layers - 1):
            convs.append(SimpleGraphConv(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(convs)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = node_features
        for conv in self.convs:
            hidden = F.relu(conv(hidden, adjacency))
        return hidden


class GraphActorCritic(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_capabilities: int):
        super().__init__()
        self.encoder = GraphEncoder(node_dim=node_dim, hidden_dim=hidden_dim)
        self.host_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.capability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_capabilities),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor):
        node_embeddings = self.encoder(node_features, adjacency)
        host_logits = self.host_head(node_embeddings).squeeze(-1)
        graph_embedding = node_embeddings.mean(dim=0)
        capability_logits = self.capability_head(graph_embedding)
        value = self.value_head(graph_embedding).squeeze(-1)
        return host_logits, capability_logits, value

    def sample_action(self, node_features: torch.Tensor, adjacency: torch.Tensor, actions_per_node: int):
        host_logits, capability_logits, value = self(node_features, adjacency)
        host_dist = torch.distributions.Categorical(logits=host_logits)
        cap_dist = torch.distributions.Categorical(logits=capability_logits)
        host = host_dist.sample()
        capability = cap_dist.sample()
        action = host * actions_per_node + capability
        log_prob = host_dist.log_prob(host) + cap_dist.log_prob(capability)
        return action.item(), log_prob, value

    def greedy_action(self, node_features: torch.Tensor, adjacency: torch.Tensor, actions_per_node: int):
        host_logits, capability_logits, value = self(node_features, adjacency)
        host = torch.argmax(host_logits)
        capability = torch.argmax(capability_logits)
        action = host * actions_per_node + capability
        return action.item(), value


def build_graph_tensors(graph_snapshot, device="cpu"):
    node_features = torch.tensor(graph_snapshot["node_features"], dtype=torch.float32, device=device)
    adjacency = torch.tensor(graph_snapshot["adjacency"], dtype=torch.float32, device=device)
    return node_features, adjacency