import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask, src_key_padding_mask, pos):
        q = k = src if pos is None else src + pos
        output = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(output)
        src = self.norm1(src)
        output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        output = src + self.dropout2(output)
        output = self.norm2(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout, enc_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(enc_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos,
                           src_key_padding_mask=src_key_padding_mask)
        return output
