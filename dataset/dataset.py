import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.transform_utils import make_transforms


def text_to_features(text, max_len, tokenizer):
    """
    Convert text to features
    Appending [CLS] and [SEP] token
    """
    tokens = []
    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > max_len - 2:
        tokenized_text = tokenized_text[0:(max_len - 2)]
    tokens.append('[CLS]')
    tokens.extend(tokenized_text)
    tokens.append('[SEP]')
    tokens_mask = [1] * len(tokens)
    tokens_id = tokenizer.convert_tokens_to_ids(tokens)
    while len(tokens_id) < max_len:
        tokens_id.append(0)
        tokens_mask.append(0)

    return tokens, np.array(tokens_id, dtype=int), np.array(tokens_mask, dtype=int)


class TransVGDataSet(data.Dataset):
    SUPPORTED_DATASETS = {
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        }
    }

    def __init__(self, args, split='train'):
        self.images = []
        self.data_root = args.data_root
        self.split_root = args.split_root
        self.dataset = args.dataset
        self.max_len = args.max_len
        self.transform = make_transforms(args, split)
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # using augmentation
        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        # dataset path
        self.dataset_root = os.path.join(self.data_root, 'other')
        self.image_dir = os.path.join(self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
        self.split_dir = os.path.join(self.dataset_root, 'splits')

        dataset_path = os.path.join(self.split_root, self.dataset)
        splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = os.path.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def __len__(self):
        return len(self.images)

    def __get_raw__(self, idx):
        img_file, _, box, text, attr = self.images[idx]
        # adapt box format
        box = np.array(box, dtype=int)
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        box = torch.tensor(box).float()
        # load image
        img_path = os.path.join(self.image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        # convert text to lowercase
        text = text.lower()
        return img, text, box

    def __getitem__(self, idx):
        img, text, box = self.__get_raw__(idx)
        input_data = {
            'img': img,
            'box': box,
            'text': text
        }
        input_data = self.transform(input_data)
        img = input_data['img']
        img_mask = input_data['mask']
        box = input_data['box']
        text = input_data['text']

        # get text embedding
        _, tokens_id, tokens_mask = text_to_features(text=text, max_len=self.max_len, tokenizer=self.tokenizer)

        return img, np.array(img_mask), tokens_id, tokens_mask, np.array(box, dtype=np.float32)


def collate_fn(raw_batch):
    raw_batch = list(zip(*raw_batch))
    img = torch.stack(raw_batch[0])
    img_mask = torch.tensor(raw_batch[1])
    word_id = torch.tensor(raw_batch[2])
    word_mask = torch.tensor(raw_batch[3])
    bbox = torch.tensor(raw_batch[4])
    batch = [img, img_mask, word_id, word_mask, bbox]
    return tuple(batch)
