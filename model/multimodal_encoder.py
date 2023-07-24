import torch.nn as nn

from .transformer_encoder import TransformerEncoder


class MultimodalEncoder(nn.Module):
    """
    Multimodal Encoder with 6 layers transformer encoder
    """
    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(hidden_dim=args.vl_hidden_dim, num_heads=args.vl_nheads,
                                          ffn_dim=args.vl_ffn_dim, enc_layers=args.vl_enc_layers,
                                          dropout=args.vl_dropout)
        self.hidden_dim = args.vl_hidden_dim
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src, mask, pos):
        return self.encoder(src=src, src_key_padding_mask=mask, pos=pos)
