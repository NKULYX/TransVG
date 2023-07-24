import time
import json
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

import utils.misc_utils as utils
import utils.metric_utils as metric_utils


@torch.no_grad()
def validate(args, model, data_loader):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img, img_mask, text, text_mask, target = batch
        batch_size = img.size(0)
        # copy to GPU
        img = img.to(args.device)
        img_mask = img_mask.to(args.device)
        text = text.to(args.device)
        text_mask = text_mask.to(args.device)
        target = target.to(args.device)

        pred_boxes = model(img, img_mask, text, text_mask)
        miou, accu = metric_utils.eval_val(pred_boxes, target)

        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def train_one_epoch(args, model, data_loader, optimizer, scheduler, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img, img_mask, text, text_mask, target = batch

        # copy to GPU
        img = img.to(args.device)
        img_mask = img_mask.to(args.device)
        text = text.to(args.device)
        text_mask = text_mask.to(args.device)
        target = target.to(args.device)

        # model forward
        output = model(img, img_mask, text, text_mask)

        loss_dict = metric_utils.loss(output, target)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args, model, sampler, train_data_loader, val_data_loader, optimizer, scheduler):
    output_dir = Path(args.output_dir)
    model.train()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, train_data_loader, optimizer, scheduler, epoch)
        val_stats = validate(args, model, val_data_loader)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}
        if args.output_dir:
            # output log
            if utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            # save checkpoint
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if val_stats['accu'] > best_acc:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_acc = val_stats['accu']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)


@torch.no_grad()
def evaluate(args, model, data_loader):
    output_dir = Path(args.output_dir)
    model.eval()
    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img, img_mask, text, text_mask, target = batch
        # copy to GPU
        img = img.to(args.device)
        img_mask = img_mask.to(args.device)
        text = text.to(args.device)
        text_mask = text_mask.to(args.device)
        target = target.to(args.device)

        output = model(img, img_mask, text, text_mask)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    acc_num = metric_utils.eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([acc_num, total_num]).to(args.device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    if utils.is_main_process():
        log_stats = {'test_model:': args.eval_model,
                     '%s_set_accuracy' % args.eval_set: accuracy}
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "eval_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
