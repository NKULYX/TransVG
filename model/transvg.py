import torch
import torch.nn as nn
import torch.nn.functional as F

from .textual_encoder import TextualEncoder
from .visual_encoder import VisualEncoder
from .multimodal_encoder import MultimodalEncoder


class BoxPredictor(nn.Module):
    """
    Predict the box coordinates (3 layers MLP)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.num_layers = num_layers
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class TransVG(nn.Module):
    """
    TransVG model
    Textual encoder : 12 layers Bert
    Visual encoder : Detr architecture encoder with resnet backbone and 6 layers transformer encoder
    Multimodal encoder : 6 layers transformer encoder
    """
    def __init__(self, args):
        super(TransVG, self).__init__()
        self.hidden_dim = args.vl_hidden_dim
        self.path_size = 16 if args.dilation else 32
        self.visual_tokens = int((args.imsize / self.path_size) ** 2)
        self.textual_tokens = args.max_len
        self.pos_embed = nn.Embedding(self.visual_tokens + self.textual_tokens + 1, self.hidden_dim)
        self.box_token = nn.Embedding(1, self.hidden_dim)

        self.textual_encoder = TextualEncoder(args)
        self.visual_encoder = VisualEncoder(args)
        self.textual_projector = nn.Linear(self.textual_encoder.hidden_dim, self.hidden_dim)
        self.visual_projector = nn.Linear(self.visual_encoder.hidden_dim, self.hidden_dim)
        self.multimodal_encoder = MultimodalEncoder(args)
        self.box_predictor = BoxPredictor(self.hidden_dim, self.hidden_dim, 4, 3)

    def forward(self, img, img_mask, text, text_mask):
        B = img.shape[0]
        # get image features
        img_feature, img_mask = self.visual_encoder(img, img_mask)
        img_feature = self.visual_projector(img_feature)
        # get text features
        text_feature, text_mask = self.textual_encoder(text, text_mask)
        text_feature = self.textual_projector(text_feature)
        text_feature = text_feature.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)
        # get box features
        box_feature = self.box_token.weight.unsqueeze(1).repeat(1, B, 1)
        box_mask = torch.zeros((B, 1)).to(box_feature.device).to(torch.bool)
        # concatenate all features
        multimodal_feature = torch.cat([box_feature, text_feature, img_feature], dim=0)
        multimodal_mask = torch.cat([box_mask, text_mask, img_mask], dim=1)
        multimodal_pos = self.pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        multimodal_feature = self.multimodal_encoder(multimodal_feature, multimodal_mask, multimodal_pos)
        # get the target feature (the box token embedding)
        target_feature = multimodal_feature[0]
        # make prediction of the box
        predicted_box = self.box_predictor(target_feature).sigmoid()
        return predicted_box

