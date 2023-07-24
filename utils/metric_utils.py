import torch
import torch.nn.functional as F

from utils.box_utils import bbox_iou, xywh2xyxy, generalized_box_iou


def loss(batch_pred, batch_target):
    """
    Compute the losses related to the bounding boxes,
    including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / batch_size
    losses['loss_giou'] = loss_giou.sum() / batch_size

    return losses


def eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    acc = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, acc

def eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    acc_num = torch.sum(iou >= 0.5)

    return acc_num
