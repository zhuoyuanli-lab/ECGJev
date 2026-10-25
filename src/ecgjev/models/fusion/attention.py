from torch import nn


class FusionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, queries, features, padding_mask):
        hidden = self.norm_kv(features)
        attended, weights = self.attn(self.norm_q(queries), hidden, hidden,
                                      key_padding_mask=padding_mask, need_weights=True,
                                      average_attn_weights=False)
        queries = queries + attended
        return queries + self.ffn(self.norm_f(queries)), weights
