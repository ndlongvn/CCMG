"""Graph encoders, edge-aware message passing, and omics attention fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINEConv,
    GraphNorm,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_edge


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.ReLU,
        dropout=0.0,
        norm=None,
        init_type="xavier_uniform",
    ):
        """
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list of int): List of hidden layer sizes
            output_dim (int): Dimension of output features
            activation (nn.Module): Activation function class (e.g., nn.ReLU)
            dropout (float): Dropout rate (default: 0.0)
            norm (str or None): Normalization type ('batch' for BatchNorm1d, None for no normalization)
            init_type (str): Weight initialization method
                             ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'normal')
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.init_type = init_type

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_linear_(module, self.init_type)

    def forward(self, x):
        return self.network(x)


# ---- helpers ---------------------------------------------------------------


def init_linear_(lin: nn.Linear, init_type: str = "xavier_uniform"):
    if init_type == "xavier_uniform":
        nn.init.xavier_uniform_(lin.weight)
    elif init_type == "xavier_normal":
        nn.init.xavier_normal_(lin.weight)
    elif init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
    elif init_type == "kaiming_normal":
        nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
    elif init_type == "normal":
        nn.init.normal_(lin.weight, mean=0.0, std=0.02)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


# ---- GIN (GINE) wrapper ----------------------------------------------------


class GINWrapper(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        hidden_dim=None,
        bias=True,
        init_type: str = "xavier_uniform",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        # MLP used by GINEConv for message transform
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.ReLU(),
        )
        self.conv1 = GINEConv(self.mlp, edge_dim=edge_dim)
        self.init_type = init_type
        self._init_weights()

    def _init_weights(self):
        # Re-init all Linear layers inside the MLP
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                init_linear_(m, self.init_type)
        # Reset conv params (GINEConv has .reset_parameters())
        if hasattr(self.conv1, "reset_parameters"):
            self.conv1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.conv1(x, edge_index, edge_attr).relu()


# ---- GCN wrapper -----------------------------------------------------------


class GCNWrapper(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        hidden_dim=None,
        bias=True,
        init_type: str = "xavier_uniform",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.edge_mlp = nn.Linear(edge_dim, input_dim, bias=bias)
        self.conv1 = GCNConv(input_dim, hidden_dim, bias=bias)
        self.init_type = init_type
        self._init_weights()

    def _init_weights(self):
        init_linear_(self.edge_mlp, self.init_type)
        if hasattr(self.conv1, "reset_parameters"):
            self.conv1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_mlp(edge_attr)
        _, col = edge_index  # (2, E)
        # aggregate edge embeddings to target nodes (col)
        x = x + torch.zeros_like(x).scatter_add_(
            0, col.view(-1, 1).expand_as(edge_embedding), edge_embedding
        )
        return self.conv1(x, edge_index).relu()


# ---- Transformer wrapper ---------------------------------------------------


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
        num_heads: int = 1,
        bias=True,
        init_type: str = "xavier_uniform",
    ):
        super().__init__()
        self.gnn = TransformerConv(
            in_channels,
            out_channels,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            bias=bias,
        )
        self.init_type = init_type
        self._init_weights()

    def _init_weights(self):
        if hasattr(self.gnn, "reset_parameters"):
            self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.gnn(x, edge_index, edge_attr)


# ---- SAGE wrapper ---------------------------------------------------


class EdgeSAGEConv(MessagePassing):
    """
    A light edge-aware SAGEConv:
      m_ij = x_j + W_e * e_ij   (if edge_attr provided)
      h_i  = W_self * x_i + W_neigh * mean_j( m_ij )

    This keeps GraphSAGE flavor but lets edges shift neighbor messages.
    """

    def __init__(self, in_channels, out_channels, edge_dim=None, bias=True):
        super().__init__(aggr="mean")  # mean aggregation like standard SAGE
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_edge = (
            nn.Linear(edge_dim, in_channels, bias=False)
            if edge_dim and edge_dim > 0
            else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        init_linear_(self.lin_self)
        init_linear_(self.lin_neigh)
        if self.lin_edge is not None:
            init_linear_(self.lin_edge)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, F], edge_index: [2, E], edge_attr: [E, Fe] or None
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: neighbor features [E, F]
        if self.lin_edge is not None and edge_attr is not None:
            return x_j + self.lin_edge(edge_attr)
        return x_j

    def update(self, aggr_out, x):
        # aggr_out: mean of messages; x: central features
        out = self.lin_self(x) + self.lin_neigh(aggr_out)
        return out


class GraphSAGEWrapper(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim=None,
        hidden_dim=None,
        bias=True,
        init_type: str = "xavier_uniform",
    ):
        """
        If edge_dim is None or 0 -> use standard SAGEConv.
        If edge_dim > 0         -> use EdgeSAGEConv (edge-aware).
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.init_type = init_type
        self.pre = False

        if input_dim != hidden_dim:

            self.pre = True
            # Match the hidden width before message passing.
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=bias),
                nn.ReLU(),
            )

        # Choose conv
        if edge_dim is None or edge_dim == 0:
            self.conv1 = SAGEConv(hidden_dim, output_dim, bias=bias)
        else:
            self.conv1 = EdgeSAGEConv(
                hidden_dim, output_dim, edge_dim=edge_dim, bias=bias
            )

        self._init_weights()

    def _init_weights(self):
        # reset conv params
        if hasattr(self.conv1, "reset_parameters"):
            self.conv1.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: [N, input_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim] or None
        """
        if self.pre:
            x = self.mlp(x)
        # SAGEConv ignores edge_attr; EdgeSAGEConv consumes it
        if isinstance(self.conv1, EdgeSAGEConv):
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        return torch.relu(x)


# ---- GNN Encoder -----------------------------------------------------------


class GNN_Encoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        gflayer,
        hidden_dim,
        num_layer,
        dropedge_p: float = 0.0,
        force_undirected: bool = False,
        drop_ratio=0.1,
        init_type: str = "xavier_uniform",
        bias=True,
        gat_heads: int = 1,
        trans_heads: int = 1,
        trans_dropout: float = 0.0,
    ):
        """
        gflayer: one of {"GIN", "GCN", "GAT", "GTN", "GSAGE"}
        - GAT preserves the original variant without learned edge features
        - GTN uses TransformerConv
        """
        super().__init__()

        if num_layer < 1:
            raise ValueError("num_layer must be positive.")
        if gflayer == "GAT" and gat_heads != 1:
            raise ValueError(
                "This encoder requires gat_heads=1 for residual dimensions."
            )
        if gflayer == "GTN" and trans_heads != 1:
            raise ValueError(
                "This encoder requires trans_heads=1 for residual dimensions."
            )
        self.res = node_dim == hidden_dim
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.init_type = init_type
        self.gflayer = gflayer
        self.dropedge_virtual_p = dropedge_p
        self.force_undirected = force_undirected

        # Stacks of graph layers
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            in_dim = node_dim if layer == 0 else hidden_dim

            if gflayer == "GIN":
                self.gnns.append(
                    GINWrapper(
                        in_dim,
                        hidden_dim,
                        edge_dim,
                        hidden_dim=None,
                        bias=bias,
                        init_type=init_type,
                    )
                )
            elif gflayer == "GCN":
                self.gnns.append(
                    GCNWrapper(
                        in_dim,
                        hidden_dim,
                        edge_dim,
                        hidden_dim=None,
                        bias=bias,
                        init_type=init_type,
                    )
                )
            elif gflayer == "GAT":
                # Preserve the original GAT variant, which ignores edge attributes.
                self.gnns.append(
                    GATConv(in_dim, hidden_dim, heads=gat_heads, bias=bias)
                )
            elif gflayer == "GTN":
                self.gnns.append(
                    TransformerWrapper(
                        in_dim,
                        hidden_dim,
                        edge_dim,
                        dropout=trans_dropout,
                        num_heads=trans_heads,
                        bias=bias,
                        init_type=init_type,
                    )
                )
            elif gflayer == "GSAGE":
                self.gnns.append(
                    GraphSAGEWrapper(
                        in_dim, hidden_dim, edge_dim, bias=bias, init_type=init_type
                    )
                )
            else:
                raise ValueError(f"Unknown gflayer: {gflayer}")

        # Norm per layer
        self.batch_norms = nn.ModuleList(
            [GraphNorm(hidden_dim) for _ in range(num_layer)]
        )

        # Preserve the original initialization order across nested modules.
        self._init_weights_extra()

    def _init_weights_extra(self):
        # Reinitialize registered linear layers and convolution parameters.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Preserve the original initialization order for checkpoint parity.
                init_linear_(m, self.init_type)
            # For PyG convs created directly here (e.g., GATConv), call their reset:
            if hasattr(m, "reset_parameters") and not isinstance(m, (nn.BatchNorm1d,)):
                # Most convs implement reset_parameters
                m.reset_parameters()

    def _virtual_node_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask of shape [N] where True marks the virtual node of each graph.
        Assumes the virtual node is the last node within each graph segment.
        """
        N = batch.size(0)
        num_graphs = (
            int(batch[-1].item()) + 1
        )  # batch is non-decreasing: last value + 1
        counts = torch.bincount(batch, minlength=num_graphs)  # [G]
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts[:-1], dim=0)  # [G]
        last_idx = starts + counts - 1  # [G]
        is_virtual = torch.zeros(N, dtype=torch.bool, device=batch.device)  # [N]
        is_virtual[last_idx] = True
        return is_virtual

    def forward(self, x, edge_index, edge_attr, batch=None):
        h_list = [x]
        for layer in range(self.num_layer):

            # Each layer samples virtual-edge dropout independently.
            ei, ea = edge_index, edge_attr

            if self.training and self.dropedge_virtual_p > 0.0:
                if batch is None:
                    raise KeyError(
                        f"batch is None with edge drop value {self.dropedge_virtual_p}"
                    )

                is_virtual = self._virtual_node_mask(batch)  # [N]

                row, col = ei
                virt_edge_mask = (
                    is_virtual[row] | is_virtual[col]
                )  # edges touching virtual nodes
                real_edge_mask = ~virt_edge_mask

                # split
                real_ei, real_ea = ei[:, real_edge_mask], ea[real_edge_mask]
                virt_ei, virt_ea = ei[:, virt_edge_mask], ea[virt_edge_mask]

                # drop among virtual edges only
                virt_ei, retained_edges = dropout_edge(
                    virt_ei,
                    p=self.dropedge_virtual_p,
                    force_undirected=self.force_undirected,
                    training=True,
                )

                virt_ea = virt_ea[retained_edges]

                # recombine
                if virt_ei.numel() == 0:
                    ei, ea = real_ei, real_ea
                else:
                    ei = torch.cat([real_ei, virt_ei], dim=1)
                    ea = torch.cat([real_ea, virt_ea], dim=0)

            # pass the possibly-dropped graph to this layer
            h = self.gnns[layer](h_list[layer], ei, ea)
            h = (
                self.batch_norms[layer](h, batch)
                if batch is not None
                else self.batch_norms[layer](h)
            )

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.elu(h), self.drop_ratio, training=self.training)

            # residual
            use_res = self.res if layer == 0 else True
            if use_res:
                h = h + h_list[-1]

            h_list.append(h)
        return h_list[-1]


# ---- Cell Fusion -----------------------------------------------------------


class CellFusionMHA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_cell: int = 3,
        model_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_cls: bool = False,
        agg: str = "cls",
        out_dim: int = None,
        init_type: str = "xavier_uniform",
    ):
        super().__init__()
        model_dim = model_dim or dim
        # out_dim is a legacy argument; the fusion output width is model_dim.
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")

        self.in_proj = nn.Linear(dim, model_dim)
        self.mod_embed = nn.Parameter(torch.zeros(1, num_cell, model_dim))
        nn.init.trunc_normal_(self.mod_embed, std=0.02)

        self.use_cls = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.mha = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ln1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(model_dim)

        self.agg = agg
        self.init_type = init_type

        self._init_weights()

    def _init_weights(self):
        # init all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear_(m, self.init_type)

    def forward(self, v_cell, return_att_score: bool = False):
        """Fuse modality vectors; optionally return the legacy modality scores.

        With a CLS token, scores are its attention weights toward modalities.
        The default mean aggregation returns normalized token L2 norms instead.
        """
        V = torch.stack(v_cell, dim=1)
        V = self.in_proj(V) + self.mod_embed

        if self.use_cls:
            B = V.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            X = torch.cat([cls, V], dim=1)
        else:
            X = V

        attn_out, attn_w = self.mha(X, X, X, need_weights=True)
        X = self.ln1(X + attn_out)
        X = self.ln2(X + self.ffn(X))

        if self.use_cls and self.agg == "cls":
            fused = X[:, 0, :]
            if return_att_score:
                alpha = attn_w[:, 0, 1:]
        else:
            start = 1 if self.use_cls else 0
            fused = X[:, start:, :].mean(dim=1)
            if return_att_score:
                alpha = X[:, start:, :].norm(p=2, dim=-1)
                alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        if return_att_score:
            return fused, alpha
        return fused
