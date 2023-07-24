import argparse
import datetime
import random
import time

import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc_utils as utils
from model.transvg import TransVG
from dataset.dataset import TransVGDataSet, collate_fn
from module.framework import train, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_transformer', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--ffn_dim', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_ffn_dim', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evaluation options
    parser.add_argument('--eval_model', default='', type=str)

    return parser


def set_seed(seed):
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    set_seed(args.seed)

    # build model
    model = TransVG(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    backbone_param = [p for n, p in model_without_ddp.named_parameters() if
                      (("visual_encoder" in n) and ("backbone" in n) and p.requires_grad)]
    transformer_param = [p for n, p in model_without_ddp.named_parameters() if
                         (("visual_encoder" in n) and ("backbone" not in n) and p.requires_grad)]
    bert_param = [p for n, p in model_without_ddp.named_parameters() if (("textual_encoder" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if
                  (("visual_encoder" not in n) and ("textual_encoder" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                  {"params": backbone_param, "lr": args.lr_backbone},
                  {"params": transformer_param, "lr": args.lr_transformer},
                  {"params": bert_param, "lr": args.lr_bert}]

    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # build dataset
    dataset_test = TransVGDataSet(args, 'test') if args.eval else None
    dataset_train = TransVGDataSet(args, 'train') if not args.eval else None
    dataset_val = TransVGDataSet(args, 'val') if not args.eval else None

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True) if dataset_train is not None else None
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if dataset_val is not None else None
        sampler_test = DistributedSampler(dataset_test, shuffle=False) if dataset_test is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train) if dataset_train is not None else None
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

    # build dataloader
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False,
                                  collate_fn=collate_fn, num_workers=args.num_workers) if args.eval else None
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size,
                                                        drop_last=True) if not args.eval else None
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers) if not args.eval else None
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers) if not args.eval else None

    # load checkpoints
    if args.eval:
        checkpoint = torch.load(args.eval_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
    else:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
        elif args.detr_model is not None:
            checkpoint = torch.load(args.detr_model, map_location='cpu')
            missing_keys, unexpected_keys = model_without_ddp.visual_encoder.load_state_dict(checkpoint['model'],
                                                                                             strict=False)
            print('Missing keys when loading detr model:')
            print(missing_keys)

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        if args.eval:
            with (output_dir / "eval_log.txt").open("a") as f:
                f.write(str(args) + "\n")
        else:
            with (output_dir / "train_log.txt").open("a") as f:
                f.write(str(args) + "\n")

    if not args.eval:
        print("Start training")
        start_time = time.time()
        train(args, model, sampler_train, data_loader_train, data_loader_val, optimizer, lr_scheduler)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    print("Start evaluating")
    start_time = time.time()
    evaluate(args, model, data_loader_test)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
