import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .transformer_encoder import TransformerEncoder


class ResNet(nn.Module):
    """
    ResNet backbone: using the output of layer4 as final output
    """
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.body = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation],
            pretrained=False
        )
        self.body.fc = None
        self.num_channels = 2048
        # only fine-tune layer1
        for name, parameter in self.body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

    def forward(self, img, img_mask):
        x = img
        out = None
        # get the output of layer4
        for name, module in self.body.named_children():
            x = module(x)
            if name == 'layer4':
                out = x
                break
        img_mask = F.interpolate(img_mask[None].float(), size=out.shape[-2:]).to(torch.bool)[0]
        return out, img_mask


class VisualPosEmbed(nn.Module):
    """
    Positional encoding for visual features using in 'Attention is All You Need'
    """
    def __init__(self, args):
        super().__init__()
        self.num_pos_feats = args.hidden_dim // 2
        self.temperature = 10000
        self.normalize = True
        self.scale = 2 * math.pi

    def forward(self, x, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Backbone(nn.Sequential):
    """
    ResNet backbone with positional encoding
    """
    def __init__(self, args):
        backbone = ResNet(args)
        position_embedding = VisualPosEmbed(args)
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, x):
        emb, mask = self[0](x['img'], x['img_mask'])
        pos = self[1](emb, mask)
        return emb, mask, pos


class VitEncoder(nn.Module):
    """
    Visual transformer encoder of 6 layers
    """
    def __init__(self, args):
        super(VitEncoder, self).__init__()
        self.encoder = TransformerEncoder(hidden_dim=args.hidden_dim, num_heads=args.nheads,
                                          ffn_dim=args.ffn_dim, dropout=args.dropout,
                                          enc_layers=args.enc_layers)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src, mask, pos):
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        output = self.encoder(src=src, src_key_padding_mask=mask, pos=pos)
        return output, mask


class VisualEncoder(nn.Module):
    """
    Visual Encoder: ResNet architecture backbone + Transformer
    """
    def __init__(self, args):
        super(VisualEncoder, self).__init__()
        self.backbone = Backbone(args)
        self.transformer = VitEncoder(args)
        self.hidden_dim = args.hidden_dim
        self.input_proj = nn.Conv2d(in_channels=self.backbone.num_channels, out_channels=self.hidden_dim,
                                    kernel_size=1)
        if not args.lr_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        if not args.lr_transformer:
            for param in self.transformer.parameters():
                param.requires_grad_(False)

    def forward(self, img, img_mask):
        x = {
            'img': img,
            'img_mask': img_mask
        }
        emb, mask, pos = self.backbone(x)
        emb = self.input_proj(emb)
        visual_feature, mask = self.transformer(emb, mask, pos)
        return visual_feature, mask
