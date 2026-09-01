"""Transformer recommender with KG regularization.

Same architecture as Model 2, but training adds a KG loss term that pulls
KG-connected books together in the item-embedding space:

  total_loss = MLM_loss + KG_LAMBDA * KG_loss

  KG_loss = mean over sampled edges (i,j) of (1 - cosine_sim(item_emb[i], item_emb[j]))

The model itself is identical to Model 2 -- the KG only changes the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRecommenderKG(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, n_input_bins, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.item_emb = nn.Embedding(ntoken, d_model, padding_idx=pad_id)
        self.value_emb = nn.Embedding(n_input_bins, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, n_input_bins)

    def forward(self, genes, values, src_key_padding_mask):
        x = self.item_emb(genes) + self.value_emb(values)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        cell_emb = x[:, 0, :]
        mlm_output = self.mlm_head(x)
        return {"cell_emb": cell_emb, "mlm_output": mlm_output}

    def kg_loss(self, edge_i_tokens, edge_j_tokens):
        """Cosine-similarity regularization on item embeddings for KG edges.
        edge_i_tokens, edge_j_tokens: (B,) token ids of connected books."""
        ei = self.item_emb(edge_i_tokens)
        ej = self.item_emb(edge_j_tokens)
        cos = F.cosine_similarity(ei, ej, dim=1)
        return (1.0 - cos).mean()
