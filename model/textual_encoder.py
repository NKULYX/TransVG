import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


class TextualEncoder(nn.Module):
    """
    Textual Encoder : 12 layers of Bert encoder
    """
    def __init__(self, args):
        super(TextualEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not args.lr_bert > 0:
            for param in self.bert.parameters():
                param.requires_grad_(False)
        self.encoder_num = args.bert_enc_num
        self.hidden_dim = 768

    def forward(self, text_id, text_mask):
        output, _ = self.bert(input_ids=text_id, token_type_ids=None, attention_mask=text_mask)
        textual_feature = output[self.encoder_num - 1]
        mask = text_mask.to(torch.bool)
        mask = ~mask
        return textual_feature, mask

