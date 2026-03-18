# Generalization Roadmap

This document records the next implementation steps for moving from
environment-specific policies to network-family generalization.

## 1. Network Generator Design

### Objective

Replace hand-crafted test networks such as X, Y, and Z with a network family
generator that samples unseen tasks from a broader distribution.

### Design Goals

- Sample a new network graph at every episode reset.
- Decouple learning from fixed node indices.
- Support both training-time randomization and test-time held-out distributions.
- Preserve a stable action semantics through capability abstraction.

### Task Factors To Randomize

- Number of subnets.
- Number of hosts per subnet.
- Reachability graph between subnets.
- Gate constraints for lateral movement.
- Goal host location.
- Initial foothold location.
- Host vulnerability family assignments.
- Firewall blocks or inaccessible routes.
- Decoy hosts and dead-end branches.
- Scan noise and incomplete observations.

### Proposed Generator API

```python
from dataclasses import dataclass

@dataclass
class NetworkTask:
    subnet_sizes: list[int]
    edges: list[tuple[int, int]]
    host_subnet: list[int]
    vuln_family: list[int]
    goal_node: int
    initial_visible_subnets: list[int]
    gate_policy: dict[int, dict]
    scan_noise: float
    decoy_nodes: list[int]


class NetworkTaskGenerator:
    def sample(self, split: str = "train") -> NetworkTask:
        ...
```

### Train / Validation / Test Split Strategy

- `train`: broad randomization over topology, vuln families, and goals.
- `val`: same factor space but with different random seeds.
- `test-iid`: same generator, unseen samples.
- `test-ood-topology`: held-out topology families.
- `test-ood-security`: held-out vuln distributions or gate rules.

### Recommended Topology Families

- Linear multi-subnet chain.
- Key-node pivot chain.
- Dual-control core access.
- Hub-and-spoke enterprise network.
- Tree-structured branch office network.
- Scale-free network with central choke points.

### Anti-Overfitting Measures

- Node permutation at reset.
- Goal-node randomization.
- Variable subnet sizes.
- Random vulnerability families per host.
- Partial observability with noisy scan outputs.

### Minimal Refactor Plan

1. Introduce `NetworkTask` as a sampled latent task.
2. Make the environment consume `NetworkTask` instead of hard-coded arrays.
3. Add `split`-aware generator support.
4. Benchmark against X/Y/Z as fixed held-out tasks.


## 2. GNN Policy Design

### Objective

Replace flat MLP policies with a graph-aware policy that reasons over hosts,
subnets, reachability, and exploit capabilities.

### Why GNN

The current flat observation vector encourages memorization of node positions.
A GNN can learn permutation-robust host embeddings and support variable graph
structure more naturally.

### Proposed State Representation

- Node features:
  - discovered
  - exploited
  - vulnerability flags or family embedding
  - subnet id embedding
  - privilege level
  - decoy flag
  - goal flag
- Edge features:
  - reachable
  - requires pivot
  - blocked by firewall
- Global features:
  - current topology type
  - task embedding
  - scan uncertainty level

### Proposed Policy Structure

```text
Node Features + Edge Index
        |
        v
  Graph Encoder (2-4 message passing layers)
        |
        +--> Host Selector Head
        |
        +--> Capability Selector Head
        |
        +--> Value Head
```

### Recommended Two-Stage Action Factorization

- Stage 1: choose target host.
- Stage 2: choose capability for that host.

This is a practical form of hierarchical action without requiring a fully
separate high-level and low-level controller.

### Concrete Model Sketch

```python
class GraphActorCritic(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_capabilities):
        super().__init__()
        self.encoder = GraphEncoder(node_dim, hidden_dim)
        self.host_head = nn.Linear(hidden_dim, 1)
        self.cap_head = nn.Linear(hidden_dim, num_capabilities)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index):
        host_embeddings = self.encoder(node_features, edge_index)
        host_logits = self.host_head(host_embeddings).squeeze(-1)
        pooled = host_embeddings.mean(dim=0)
        cap_logits = self.cap_head(pooled)
        value = self.value_head(pooled)
        return host_logits, cap_logits, value
```

### Action Assembly

```python
host = sample(host_logits)
capability = sample(cap_logits)
action = host * actions_per_node + capability
```

### Training Recommendation

- Start with PPO using the factorized host/capability action space.
- Compare against the current MLP PPO baseline.
- Evaluate on held-out graph families and node permutations.
- Track zero-shot and few-shot performance separately.

### Expected Benefit

- Better invariance to node ordering.
- More stable transfer across topology changes.
- Cleaner path toward variable-size networks.


## 3. Paper Draft: Future Work / Method Extension

### Draft Paragraph

The current study demonstrates that strong performance in a fixed penetration-
testing environment does not imply out-of-distribution robustness. Although the
proposed environment randomization and topology-aware observations improved
adaptation under few-shot fine-tuning, zero-shot transfer to structurally novel
networks remained limited. A natural next step is to replace the current fixed-
dimension state encoder with a graph-based policy architecture that operates on
host-level features and network reachability relations directly. Such a design
would reduce dependence on host ordering and provide a more suitable inductive
bias for enterprise-scale attack graphs. In parallel, future work should extend
the current handcrafted test cases into a task generator that samples network
topologies, vulnerability distributions, pivot constraints, and goal placements
from a broader distribution. This would enable training and evaluation over
network families rather than individual environments.

### Draft Paragraph: Capability Abstraction

Another important extension concerns exploit execution. In the current setup,
actions are abstracted as exploit families rather than bound to a specific tool.
This abstraction should be preserved and expanded in future work so that a
single high-level capability can be backed by a Metasploit module, a custom
proof-of-concept script, or an external security tool. Such a capability layer
would allow the policy to reason over attack semantics while remaining agnostic
to the concrete exploitation backend, thereby reducing dependence on the
coverage of any single exploit database.

### Draft Paragraph: Adaptation Benchmark

Beyond zero-shot evaluation, future studies should benchmark rapid adaptation
explicitly. In operational settings, an autonomous penetration-testing agent is
more likely to encounter partially familiar environments than exact replicas of
its training environment. Accordingly, a practical benchmark should measure not
only success rate on unseen networks, but also the number of episodes or update
steps required to recover high performance under few-shot adaptation. This would
provide a more realistic assessment of deployability in real-world enterprise
networks.

### Suggested Paper Section Outline

1. Limitation of fixed-environment training.
2. Randomized task-family training.
3. Graph-based topology-aware policy.
4. Capability abstraction beyond Metasploit coverage.
5. Few-shot adaptation benchmark for deployment realism.