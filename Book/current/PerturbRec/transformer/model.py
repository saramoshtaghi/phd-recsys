"""Self-contained transformer recommender (scGPT-style interface).

Forward(genes, values, src_key_padding_mask) -> {"cell_emb", "mlm_output"}.
- cell_emb = CLS-position output (the user embedding).
- mlm_output = per-position value-bin logits (for MLM self-supervision)."""
import torch.nn as nn


class TransformerRecommender(nn.Module):
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
        cell_emb = x[:, 0, :]           # CLS is always at position 0
        mlm_output = self.mlm_head(x)   # (B, L, n_bins)
        return {"cell_emb": cell_emb, "mlm_output": mlm_output}
