# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import os.path as osp
import numpy as np

from IPython import embed

from loguru import logger


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.

    This class is taken from Group-Free code.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter for hard and easy examples.
            alpha: Weighting parameter for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
        max(x, 0) - x * z + log(1 + exp(-abs(x))) in

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0) - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


'''
description:  计算生成query 的loss 
param {*} end_points
param {*} topk
return {*}
'''
def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    #!==============================================================
    # supervised_mask  = end_points['supervised_mask']
    # supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()
    #!==============================================================

    box_label_mask = end_points['box_label_mask']

    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K

    gt_center = end_points['center_label'][:, :, :3]  # B, G, 3
    gt_size = end_points['size_gts'][:, :, :3]  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]  # number if points from p++ output
    G = gt_center.shape[1]  # number of gt boxes (with padding)

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']  # B, num_points
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, K
    obj_assignment[obj_assignment < 0] = G - 1  # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxGxK

    # Find the points that lie closest to each gt centroid
    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss






'''
description:  text-guided keypoint sampling loss
param {*} data_dict
param {*} topk
param {*} args
return {*}
'''
def compute_kps_loss(data_dict, topk):
    #!===============
    # supervised_mask  = data_dict['supervised_mask']
    # supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()
    use_ref_score_loss = True
    ref_use_obj_mask= True
    #!===============

    box_label_mask = data_dict['box_label_mask']
    seed_inds = data_dict['seed_inds'].long()  # B, K
    seed_xyz = data_dict['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = data_dict['seeds_obj_cls_logits']  # B, 1, K
    gt_center = data_dict['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = data_dict['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = data_dict['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    data_dict[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    data_dict[f'points_hard_topk{topk}_neg_ratio'] = 1 - data_dict[f'points_hard_topk{topk}_pos_ratio']

    #* Compute objectness loss
    objectness_loss = 0
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss += cls_loss_src.sum() / B

    if 'kps_ref_score' in data_dict.keys() and use_ref_score_loss:
        #*====================================
        #* data_dict['point_instance_label'] : 不太一样, 这个point_instance_label 有多个 target 
        #*====================================
        # point_ref_mask = data_dict['point_ref_mask'] #* 3D SPS 每个sample 默认只有一个目标 by default

        point_ref_mask = data_dict['point_instance_label'] #! error
        point_ref_mask = (point_ref_mask!=-1)*1 #* -1 表示背景, 其他都表示referred target
        point_ref_mask = torch.gather(point_ref_mask, 1, seed_inds)

        if 'ref_query_points_sample_inds' in data_dict.keys():
            query_points_sample_inds = data_dict['query_points_sample_inds'].long()
            point_ref_mask = torch.gather(point_ref_mask, 1, query_points_sample_inds)

            if ref_use_obj_mask:

                obj_mask = torch.gather(objectness_label, 1, query_points_sample_inds)
                point_ref_mask = point_ref_mask * obj_mask
        kps_ref_score = data_dict['kps_ref_score']      # [B, 1, N]
        cls_weights = torch.ones((B, kps_ref_score.shape[-1])).cuda().float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        kps_ref_loss = criterion(kps_ref_score.view(kps_ref_score.shape[0], kps_ref_score.shape[2], 1),
                                 point_ref_mask.unsqueeze(-1), weights=cls_weights)

        objectness_loss += kps_ref_loss.sum() / B
        #!====================
        # tmp = kps_ref_loss.sum() / B
        # logger.info(f"objectness_loss:{objectness_loss},\t kps_ref_loss:{tmp}")
        # objectness_loss += tmp
        #!====================

    
    #* Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        data_dict[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss, data_dict


'''
description: 
param {*} end_points
param {*} topk
return {*}
'''
def compute_labeled_points_obj_cls_loss_hard_topk(end_points, topk):
    #!==============================================================
    supervised_mask  = end_points['supervised_mask']
    supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()
    #!==============================================================

    box_label_mask = end_points['box_label_mask']

    seed_inds = end_points['seed_inds'][supervised_inds,:].long()  # B, K
    seed_xyz = end_points['seed_xyz'][supervised_inds,:,:]  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits'][supervised_inds,:,:]  # B, 1, K

    gt_center = end_points['center_label'][:, :, :3]  # B, G, 3
    gt_size = end_points['size_gts'][:, :, :3]  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]  # number if points from p++ output
    G = gt_center.shape[1]  # number of gt boxes (with padding)

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']  # B, num_points
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, K
    obj_assignment[obj_assignment < 0] = G - 1  # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxGxK

    # Find the points that lie closest to each gt centroid
    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss







class HungarianMatcher(nn.Module):
    """
    Assign targets to predictions.

    This class is taken from MDETR and is modified for our purposes.

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        """
        Initialize matcher.

        Args:
            cost_class: relative weight of the classification error
            cost_bbox: relative weight of the L1 bounding box regression error
            cost_giou: relative weight of the giou loss of the bounding box
            soft_token: whether to use soft-token prediction
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits" (tensor): [batch_size, num_queries, num_classes]
                "pred_boxes" (tensor): [batch_size, num_queries, 6], cxcyczwhd
            targets: list (len(targets) = batch_size) of dict:
                "labels" (tensor): [num_target_boxes]
                    (where num_target_boxes is the no. of ground-truth objects)
                "boxes" (tensor): [num_target_boxes, 6], cxcyczwhd
                "positive_map" (tensor): [num_target_boxes, 256]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Notation: {B: batch_size, Q: num_queries, C: num_classes}
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  #* [B*Q, C],#* B,query_num, distribution_for_tokens 前两维铺平, 对最后一维取softmax, 
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  #* [B*Q, 6]

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.soft_token:
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1)) #* generating [BxQ, positive_map_num] 
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix, #* multiple the corresponding weights 
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        ).view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        #* scipy 的linear assignment problem solution,  也就是Hungarian match problem, 求助cost最小的 match 
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred boxes
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt boxes
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """
    Computes the loss in two steps:
        1) compute hungarian assignment between ground truth and outputs
        2) supervise each pair of matched ground-truth / prediction
    """

    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses #* ==['boxes', 'label']
        self.temperature = temperature

    def loss_labels_st(self, outputs, targets, indices, num_boxes):
        """Soft token prediction (with objectness).""" 
        #* soft token loss , 
        #* 每个query 都要预测一个 256维度的分布, 与256 个token 对应, gt就是positive_map(是一个对应的gt token 位置为1 ,其他位置为0的一个map)
        logits = outputs["pred_logits"].log_softmax(-1)  # (B, Q, 256)
        positive_map = torch.cat([t["positive_map"] for t in targets])

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        #? Labels, by default lines map to the last element, no_object
        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1 #* 匹配到最后一个表示no matched 
        target_sim[src_idx] = tgt_pos

        # Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute bbox losses. """#?  1.  gt 的bbox 指的是文本对应的bbox,     计算predicted bbox是否和这个bbox对应!
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]#? ==torch.cat([outputs['pred_boxes'][idx[0][0],idx[1][0]].view(-1,6),outputs['pred_boxes'][idx[0][1],idx[1][1]].view(-1,6)])
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        #*  
        loss_bbox = (
            F.l1_loss(
                src_boxes[..., :3], target_boxes[..., :3],
                reduction='none'
            )
            + 0.2 * F.l1_loss(
                src_boxes[..., 3:], target_boxes[..., 3:],
                reduction='none'
            )
        ) #* 位置算 L1 loss, 大小也算L1 loss但是乘了0.2 
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes #* 

        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_contrastive_align(self, outputs, targets, indices, num_boxes):
        """Compute contrastive losses between projected queries and tokens."""
        tokenized = outputs["tokenized"]

        # Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens, dim
        norm_img_emb = outputs["proj_queries"]  # B, num_queries, dim
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # B, num_queries, num_tokens

        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap[..., :logits.shape[-1]]
        positive_map = positive_map > 0

        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits

        # Loss 1: which tokens should each query match?
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)
        nb_pos = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos  # entropy of 1/nb_pos
        box_to_token_loss_ = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        # Loss 2: which queries should each token match?
        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)
        nb_pos = positive_map.sum(1) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos
        token_to_box_loss = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_contrastive_align": tot_loss / num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx #* 

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {#* 三个loss function 
            'labels': self.loss_labels_st,
            'boxes': self.loss_boxes,
            'contrastive_align': self.loss_contrastive_align
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets) #* return [B,matched_pair_number], 每个数据的格式 是[matched_id , key], key用于区分是第几个pair ,与配对的数据无关

        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes
            ))

        return losses, indices


def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5):
    """Compute Hungarian matching loss containing CE, bbox and giou."""

    #!================================================================
    DEBUG = False
    #!================================================================
    
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes #* proposal 是第一个, last 是最后一个 

    #* Ground-truth
    gt_center = end_points['center_label'][:, :, 0:3]  # B, G, 3
    gt_size = end_points['size_gts']  # (B,G,3)
    gt_labels = end_points['sem_cls_label']  # (B, G)
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
    positive_map = end_points['positive_map']
    box_label_mask = end_points['box_label_mask']
    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],#* 
            "positive_map": positive_map[b, box_label_mask[b].bool()] #* 分布? 用于计算 token 和query 的soft token loss 
        }#* 每个box 最多对应在 256个token中只能有一个响应的地方,  
        for b in range(gt_labels.shape[0]) 
    ]

    loss_ce, loss_bbox, loss_giou, loss_contrastive_align = 0, 0, 0, 0
    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']

        #* Get predicted boxes and labels, why the K equals to 256,n_class equals to 256, Q equals to 256
        pred_center = end_points[f'{prefix}center']  # B, K, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,K,3) (l,w,h)
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']  # (B, Q, n_class)
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox

        

        # Compute all the requested losses
        if DEBUG:
            losses = compute_loss_and_save_match_res_(output, target,set_criterion,end_points['scan_ids'],prefix)
        else :
            losses, _ = set_criterion(output, target)
        


        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_contrastive_align += losses['loss_contrastive_align']


    #!================================================================
    if "ref_query_points_sample_inds" in end_points.keys():
        query_points_generation_loss, end_points =compute_kps_loss(end_points, query_points_obj_topk)

    elif 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0    

    #!================================================================



    # loss
    loss = (
        8 * query_points_generation_loss
        + 1.0 / (num_decoder_layers + 1) * (
            loss_ce
            + 5 * loss_bbox
            + loss_giou
            + loss_contrastive_align
        )
    )
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_constrastive_align'] = loss_contrastive_align
    end_points['loss'] = loss
    return loss, end_points





'''
description:  根据supervised_mask 计算loss , 也就是只对labeled 的数据计算loss 
return {*}
'''
def compute_labeled_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5):
    """Compute Hungarian matching loss containing CE, bbox and giou."""
    #!================================================================
    DEBUG = False
    supervised_mask  = end_points['supervised_mask']
    supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()
    #!================================================================
    
    

    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes #* proposal 是第一个, last 是最后一个 

    #* Ground-truth
    gt_center = end_points['center_label'][:, :, 0:3]  # B, G, 3
    gt_size = end_points['size_gts']  # (B,G,3)
    gt_labels = end_points['sem_cls_label']  # (B, G)
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
    positive_map = end_points['positive_map']
    box_label_mask = end_points['box_label_mask']
    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],#* 
            "positive_map": positive_map[b, box_label_mask[b].bool()] #* 分布? 用于计算 token 和query 的soft token loss 
        }#* 每个box 最多对应在 256个token中只能有一个响应的地方,  
        for b in range(gt_labels.shape[0]) 
    ]

    loss_ce, loss_bbox, loss_giou, loss_contrastive_align = 0, 0, 0, 0
    for prefix in prefixes:
        output = {}
        #!+==========================================================
        
        
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens'][supervised_inds,:,:]
            output['proj_queries'] = end_points[f'{prefix}proj_queries'][supervised_inds,:,:]
            # output['tokenized'] = end_points['tokenized'][supervised_inds,:,:]
            output['tokenized'] = {k:v[supervised_inds,:] for k,v in end_points['tokenized'].items()}

        
        #* Get predicted boxes and labels, why the K equals to 256,n_class equals to 256, Q equals to 256
        pred_center = end_points[f'{prefix}center'][supervised_inds,:,:]  # B, K, 3
        pred_size = end_points[f'{prefix}pred_size'][supervised_inds,:,:]  # (B,K,3) (l,w,h)
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores'][supervised_inds,:,:]  # (B, Q, n_class)
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox

        # for k,v in output.items():
        #     print(v.shape)

        #!+==========================================================




        # Compute all the requested losses
        #!================================================================
        if DEBUG:
            losses = compute_loss_and_save_match_res_(output, target,set_criterion,end_points['scan_ids'],prefix)
        else :
            losses, _ = set_criterion(output, target)
        #!================================================================


        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_contrastive_align += losses['loss_contrastive_align']

    if 'seeds_obj_cls_logits' in end_points.keys():

        query_points_generation_loss = compute_labeled_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    # loss
    loss = (
        8 * query_points_generation_loss
        + 1.0 / (num_decoder_layers + 1) * (
            loss_ce
            + 5 * loss_bbox
            + loss_giou
            + loss_contrastive_align
        )
    )
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_constrastive_align'] = loss_contrastive_align
    end_points['loss'] = loss
    return loss, end_points


''' 
description:  调用set_criterion 计算loss 并且返回hungariun 匹配结果  ,并存储匹配结果
param {*} output: pred 
param {*} target :  gt
param {*} set_criterion: 计算loss 的网络
param {*} scan_ids : 数据对应的场景名字用于生成存储结果文件名
param {*} prefix : 数据对应的前缀用于生成存储结果文件名
return {*}
'''
def compute_loss_and_save_match_res_(output, target,set_criterion,scan_ids,prefix):
    with open ('vis_results/current_path.txt','r') as f :
        debug_save_path = f.read().strip()
    
    losses, indices = set_criterion(output, target) #* indices : 

    # permute predictions following indices
    batch_idx = torch.cat([
        torch.full_like(src, i) for i, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src for (src, _) in indices])
    
    pred_bb = output['pred_boxes'][batch_idx,src_idx]

    # pred_bb = [output['pred_boxes'][idx ,key.item()]  for idx, (key,key_id)  in enumerate(indices)]
    # gt_bbox = [x['boxes'][0] for x in target]

    gt_bbox = torch.cat([
        t['boxes'][i] for t, (_, i) in zip(target, indices)
    ], dim=0)

    batch_size = len(indices)

    bbox_num_per_batch =int( gt_bbox.shape[0]/batch_size)

    for idx in range(len(indices)):
        np.savetxt(osp.join(debug_save_path, 
                        '%s_gt_%sbox.txt'%(scan_ids[idx],prefix)),
                        gt_bbox[idx*bbox_num_per_batch:(idx+1)*bbox_num_per_batch].detach().cpu().numpy(),
                        fmt='%s',delimiter=' ')

        np.savetxt(osp.join(debug_save_path,'%s_pred_%sbox.txt'%(scan_ids[idx],prefix)),
                    pred_bb[idx*bbox_num_per_batch:(idx+1)*bbox_num_per_batch].detach().cpu().numpy(),
                    fmt='%s')

    return losses