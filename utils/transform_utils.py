"""
Mostly copy-paste from the original repository.
"""
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageEnhance, ImageFilter

from .box_utils import xyxy2xywh


def crop(img, box, region):
    """
    crop according to region
    """
    # crop img
    cropped_img = F.crop(img, *region)
    # adapt box
    x, y, h, w = region
    cropped_box = box - torch.tensor([y, x, y, x], dtype=torch.float32)
    cropped_box = cropped_box.reshape(2, 2)
    max_box = torch.tensor([w, h], dtype=torch.float32)
    cropped_box = torch.min(cropped_box, max_box)
    cropped_box = cropped_box.clamp(min=0).reshape(-1)
    return cropped_img, cropped_box


def resize(img, box, size, adapt_to_long=True):
    """
    resize according to size and adaptation mode
    """
    if adapt_to_long:
        ratio = float(size / float(max(img.height, img.width)))
    else:
        ratio = float(size / float(min(img.height, img.width)))
    resized_img = F.resize(img, [round(img.width * ratio), round(img.height * ratio)])
    resized_box = box * ratio
    return resized_img, resized_box


class Compose(object):
    """
    A list of transforms
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomBrighten(object):
    """
    Randomly adapt brightness according to the ratio
    """

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img):
        brighten_ratio = random.uniform(1 - self.ratio, 1 + self.ratio)
        brighten_enhancer = ImageEnhance.Brightness(img)
        brighten_img = brighten_enhancer.enhance(brighten_ratio)
        return brighten_img


class RandomContrast(object):
    """
    Randomly contrast according to the ratio
    """

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img):
        contrast_ratio = random.uniform(1 - self.ratio, 1 + self.ratio)
        contrast_enhancer = ImageEnhance.Contrast(img)
        contrast_img = contrast_enhancer.enhance(contrast_ratio)
        return contrast_img


class RandomSaturation(object):
    """
    Randomly adapt saturation according to the ratio
    """

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img):
        saturation_ratio = random.uniform(1 - self.ratio, 1 + self.ratio)
        saturation_enhancer = ImageEnhance.Color(img)
        saturation_img = saturation_enhancer.enhance(saturation_ratio)
        return saturation_img


class ColorJitter(object):
    """
    Randomly adapt color according to the ratio
    """

    def __init__(self, brighten_ratio=0.4, contrast_ratio=0.4, saturation_ratio=0.4):
        self.transforms = [RandomBrighten(brighten_ratio),
                           RandomContrast(contrast_ratio),
                           RandomSaturation(saturation_ratio)]

    def __call__(self, input_dict):
        if random.random() < 0.8:
            img = input_dict['img']
            # execute the color transforms in random order
            order = list(np.random.permutation(3))
            for i in order:
                img = self.transforms[i](img)
            input_dict['img'] = img
        return input_dict


class GaussianBlur(object):
    def __init__(self, radius=None, aug_blur=False):
        if radius is None:
            radius = [0.1, 2.0]
        self.radius = radius
        self.p = 0.5 if aug_blur else 0

    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            radius = random.uniform(self.radius[0], self.radius[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            input_dict['img'] = img
        return input_dict


class RandomHorizontalFlip(object):
    """
    Randomly flip the image
    Need to adapt the box and text
    """

    def __call__(self, input_dict):
        if random.random() < 0.5:
            img = input_dict['img']
            box = input_dict['box']
            text = input_dict['text']
            img = F.hflip(img)
            # adapt the textual description if there exists 'right' or 'left'
            text = text.replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
            # adapt the box
            box = box[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([img.width, 0, img.width, 0])
            input_dict['img'] = img
            input_dict['box'] = box
            input_dict['text'] = text
        return input_dict


class RandomResize(object):
    def __init__(self, size, adapt_to_long=True):
        self.size = size
        self.adapt_to_long = adapt_to_long

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        size = random.choice(self.size)
        resized_img, resized_box = resize(img, box, size, self.adapt_to_long)
        input_dict['img'] = resized_img
        input_dict['box'] = resized_box
        return input_dict


class RandomSizeCrop(object):
    def __init__(self, min_size, max_size, max_try=20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            # [i, j, target_w, target_h]
            region = T.RandomCrop.get_params(img, (h, w))
            box_xywh = xyxy2xywh(box)
            box_x, box_y = box_xywh[0], box_xywh[1]
            if box_x > region[0] and box_y > region[1]:
                img, box = crop(img, box, region)
                input_dict['img'] = img
                input_dict['box'] = box
                return input_dict
        return input_dict


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, input_dict):
        text = input_dict['text']
        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        # if there exists direction description in the text
        for wd in dir_words:
            if wd in text:
                return self.transforms1(input_dict)

        if random.random() < self.p:
            return self.transforms2(input_dict)
        else:
            return self.transforms1(input_dict)


class ToTensor(object):
    def __call__(self, input_dict):
        input_dict['img'] = F.to_tensor(input_dict['img'])
        return input_dict


class NormalizeAndPad(object):
    def __init__(self, mean=None, std=None, size=640, aug_translate=False):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, input_dict):
        img = input_dict['img']
        img = F.normalize(img, mean=self.mean, std=self.std)
        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h
        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)
        out_img = torch.zeros((3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()
        out_img[:, top:top + h, left:left + w] = img
        out_mask[top:top + h, left:left + w] = 0
        input_dict['img'] = out_img
        input_dict['mask'] = out_mask

        if 'box' in input_dict.keys():
            box = input_dict['box']
            box[0], box[2] = box[0] + left, box[2] + left
            box[1], box[3] = box[1] + top, box[3] + top
            h, w = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([w, h, w, h], dtype=torch.float32)
            input_dict['box'] = box

        return input_dict


def make_transforms(args, split):
    img_size = args.imsize

    if split == 'train':
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(img_size - 32 * i)
        else:
            scales = [img_size]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        return Compose([
            RandomSelect(
                RandomResize(scales),
                Compose([
                    RandomResize([400, 500, 600], adapt_to_long=False),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales),
                ]),
                p=crop_prob
            ),
            ColorJitter(0.4, 0.4, 0.4),
            GaussianBlur(aug_blur=args.aug_blur),
            RandomHorizontalFlip(),
            ToTensor(),
            NormalizeAndPad(size=img_size, aug_translate=args.aug_translate)
        ])

    if split in ['val', 'test', 'testA', 'testB']:
        return Compose([
            RandomResize([img_size]),
            ToTensor(),
            NormalizeAndPad(size=img_size),
        ])

