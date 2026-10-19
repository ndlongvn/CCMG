"""Drug graph and multi-omics fusion model for response regression."""

import torch
from torch_geometric.nn import global_mean_pool
from gnn_model import GNN_Encoder, MLP, CellFusionMHA


class GNN(torch.nn.Module):
    # raw_data, pos_embedd, nhead, and JK are legacy API arguments; they do
    # not alter the released forward computation.
    def __init__(
        self,
        num_layer,
        raw_data,
        cell_out_dim,
        node_dim,
        edge_dim,
        hidden_dim,
        pos_embedd,
        num_tasks,
        nhead,
        num_layer_2,
        hidden_dim_2=None,
        drop_ratio=0.1,
        gnn_drop_ratio=0.1,
        gnn_drop_edge_ratio=0.0,
        JK="last",
        gnn_type="GIN",
        gnn_type_interaction="GIN",
        config=None,
    ):
        super(GNN, self).__init__()
        if config is None or "omics" not in config:
            raise ValueError("config must specify the omics modalities.")
        if cell_out_dim != hidden_dim:
            raise ValueError("cell_out_dim must equal hidden_dim for cell fusion.")
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.hidden_dim = hidden_dim

        self.num_tasks = num_tasks
        self.cell_out_dim = cell_out_dim
        self.config = config
        if not hidden_dim_2:
            hidden_dim_2 = hidden_dim
        self.hidden_dim_2 = hidden_dim_2

        self.mlp_rna = MLP(
            input_dim=18451,
            hidden_dims=[cell_out_dim * 4, cell_out_dim * 2],
            output_dim=cell_out_dim,
            dropout=drop_ratio,
            norm="batch",
        )

        self.mlp_genetic = MLP(
            input_dim=735,
            hidden_dims=[cell_out_dim * 4, cell_out_dim * 2],
            output_dim=cell_out_dim,
            dropout=drop_ratio,
            norm="batch",
        )

        self.mlp_mrna = MLP(
            input_dim=20617,
            hidden_dims=[cell_out_dim * 4, cell_out_dim * 2],
            output_dim=cell_out_dim,
            dropout=drop_ratio,
            norm="batch",
        )

        num_cell = len(self.config["omics"].split("+"))

        print(f"Using omics: {self.config['omics']} ({num_cell} modalities)")

        self.mlp_cell = CellFusionMHA(hidden_dim, num_cell)

        self.drug_model_encoder = GNN_Encoder(
            node_dim,
            edge_dim,
            gnn_type,
            hidden_dim,
            num_layer=num_layer,
            drop_ratio=gnn_drop_ratio,
        )

        self.interaction_encoder = GNN_Encoder(
            hidden_dim,
            edge_dim,
            gnn_type_interaction,
            hidden_dim_2,
            num_layer=num_layer_2,
            drop_ratio=gnn_drop_ratio,
            dropedge_p=gnn_drop_edge_ratio,
        )

        # Retained for compatibility with existing state_dict checkpoints.
        # The published forward path does not use this legacy attention layer.
        self.att_motif = torch.nn.Linear(self.hidden_dim, 1)

        self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(
                self.hidden_dim * (1 + num_cell) + self.hidden_dim_2, (self.hidden_dim)
            ),
            torch.nn.ReLU(),
            torch.nn.Linear((self.hidden_dim), self.num_tasks),
        )

    def _last_node_indices(self, batch):
        # how many nodes in each graph
        sizes = torch.bincount(batch)  # (num_graphs,)
        # cumulative sizes → last index for each graph
        return torch.cumsum(sizes, 0, dtype=torch.long) - 1

    def update_super_node_rep(self, node_rep, cell_rep, batch):
        last_idx = self._last_node_indices(batch)  # (num_graphs,)
        assert last_idx.size(0) == cell_rep.size(
            0
        ), f"Size mismatch: last_idx {last_idx.size(0)}, cell_rep {cell_rep.size(0)}"
        node_rep[last_idx] = cell_rep.to(node_rep.dtype)  # in‑place write
        return node_rep

    def get_substructure_drug_cell_embeddings(self, x, batch, node_embs):
        # Boolean masks --------------------------------------------------
        sub_mask = x[:, 11].bool()
        cell_mask = x[:, 12].bool()
        virtual_mask = sub_mask | cell_mask  # drug_mask
        real_mask = ~virtual_mask

        num_graphs = int(batch.max()) + 1  # G

        # ----- pooled “real”‑node embedding (global mean per graph) -----
        pooled = global_mean_pool(node_embs[real_mask], batch[real_mask])  # (G, D)

        # ----- sub‑structures (variable length) -------------------------
        sub_embs = node_embs[sub_mask]  # (Ns, D)
        sub_batch = batch[sub_mask]  # (Ns,)
        sizes = torch.bincount(sub_batch, minlength=num_graphs)  # (G,)

        # split in one call → list of tensors per graph
        sub_list = list(torch.split(sub_embs, sizes.tolist()))  # len = G

        # ----- drug & cell nodes (exactly one each per graph) -----------
        cell_embs = node_embs[cell_mask]  # (G, D)

        return pooled, (sub_embs, sub_list, sub_batch), cell_embs

    def forward(self, data, epoch, return_feature=False, return_att_score=False):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        v_rna, v_genetic, v_mrna = data.v_rna, data.v_genetic, data.v_mrna

        encoders = {
            "expr": (self.mlp_rna, v_rna),
            "mutation": (self.mlp_genetic, v_genetic),
            "methylation": (self.mlp_mrna, v_mrna),
        }
        modalities = [name.strip() for name in self.config["omics"].split("+")]
        v_cell = []
        for name in modalities:
            if name not in encoders:
                raise ValueError(f"Unknown omics modality: {name}")
            encoder, values = encoders[name]
            v_cell.append(encoder(values))

        node_feats = self.drug_model_encoder(x, edge_index, edge_attr)
        real_mask = ~(x[:, 11].bool() | x[:, 12].bool())
        pooled_embedding = global_mean_pool(node_feats[real_mask], batch[real_mask])

        if return_att_score:
            v_cell_avg, score = self.mlp_cell(v_cell, return_att_score)
        else:
            v_cell_avg = self.mlp_cell(v_cell, return_att_score)

        node_feats_cell = self.update_super_node_rep(node_feats, v_cell_avg, batch)

        # Preserve the original whole-mini-batch GraphNorm behavior: batch is
        # deliberately omitted unless virtual-edge dropout needs its indices.
        encoder_kwargs = (
            {"batch": batch} if self.interaction_encoder.dropedge_virtual_p > 0 else {}
        )
        node_feats_fusion = self.interaction_encoder(
            node_feats_cell, edge_index, edge_attr, **encoder_kwargs
        )

        cell_embs = node_feats_fusion[x[:, 12].bool()]

        mean_embs = torch.cat([cell_embs, pooled_embedding] + v_cell, dim=-1)

        predictions = self.graph_pred_linear(mean_embs)

        if return_att_score:
            return (
                predictions,
                v_cell + [v_cell_avg, cell_embs, mean_embs],
                score,
            )

        if return_feature:
            return predictions, mean_embs

        return predictions
