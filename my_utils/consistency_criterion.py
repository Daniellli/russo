'''
Author: daniel
Date: 2023-03-27 12:02:20
LastEditTime: 2023-03-27 19:12:49
LastEditors: daniel
Description: 
FilePath: /butd_detr/my_utils/consistency_criterion.py
have a nice day
'''

 



import sys

sys.path.append("~/exp/butd_detr")
import torch

import os.path as osp
from my_utils.pc_utils import *
import torch.nn.functional as F

from data.model_util_scannet import ScannetDatasetConfig

from IPython import embed

from my_utils.utils import make_dirs,rot_x,rot_y,rot_z,points2box,box2points,focalLoss,nn_distance

from my_utils.pc_utils import * 
from loguru import logger

from models.losses import ConsistencyHungarianMatcher

import torch.distributed as dist

from models.losses import generalized_box_iou3d,box_cxcyczwhd_to_xyzxyz

DEBUG_FILT = "~/exp/butd_detr/logs/debug"

    

import wandb 

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([
        torch.full_like(src, i) for i, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx #* 

def _get_tgt_permutation_idx( indices):
    # permute targets following indices
    batch_idx = torch.cat([
        torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
    ])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class ConsistencyCriterion:


    def __init__(self,
    box_consistency_weight = 1,
    giou_consistency_weight = 1,
    soft_token_consistency_weight = 1,
    object_query_consistency_weight = 1,
    text_token_consistency_weight = 1,
    EMA_CLIP = 0.90
    ):
        """
        Args:
            end_points: dict
                {
                    center, size_scores, size_residuals_normalized, sem_cls_scores,
                    flip_x_axis, flip_y_axis, rot_mat
                }
            ema_end_points: dict
                {
                    center, size_scores, size_residuals_normalized, sem_cls_scores,
                }
        Returns:
            consistency_loss: pytorch scalar tensor
            end_points: dict
        """
        
        
        self.prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(5)] #* 6 heads + proposal 
        self.head_num  = len(self.prefixes)
        self.box_consistency_weight = box_consistency_weight
        self.giou_consistency_weight = giou_consistency_weight
        self.soft_token_consistency_weight = soft_token_consistency_weight
        self.object_query_consistency_weight = object_query_consistency_weight
        self.text_token_consistency_weight = text_token_consistency_weight
        self.EMA_CLIP = EMA_CLIP

        
    def __call__(self,end_points, ema_end_points,augmentation):

        box = torch.tensor(0.).cuda()
        box_giou = torch.tensor(0.).cuda()
        soft_token = torch.tensor(0.).cuda()
        object_query = torch.tensor(0.).cuda()
        text_token = torch.tensor(0.).cuda()

        for prefix in self.prefixes:

            student_out=parse_endpoint(end_points,prefix)
            teacher_out=parse_endpoint(ema_end_points,prefix)
            if augmentation is not None and len(augmentation.keys()) >0:
                teacher_out['pred_boxes'] = transformation_box(teacher_out['pred_boxes'],augmentation)

            center_size_consistency_loss,giou_consistency_loss,soft_token_consistency = self.get_loss(student_out, teacher_out)
            # query_consistency_loss = (1 - F.cosine_similarity(student_out['proj_queries'], teacher_out['proj_queries'])).mean()
            # token_consistency_loss = (1 - F.cosine_similarity(student_out['proj_tokens'], teacher_out['proj_tokens'])).mean()
            
            query_consistency_loss = self.compute_kl_div(student_out['proj_queries'],teacher_out['proj_queries'])
            token_consistency_loss = self.compute_kl_div(student_out['proj_tokens'],teacher_out['proj_tokens'])

            box += center_size_consistency_loss[0]
            box_giou += giou_consistency_loss[0]
            soft_token += soft_token_consistency[0]
            object_query += query_consistency_loss
            text_token += token_consistency_loss



        end_points['box_consistency_loss'] = (box / self.head_num ) * self.box_consistency_weight
        end_points['box_giou_consistency_loss'] = (box_giou / self.head_num) * self.giou_consistency_weight
        end_points['soft_token_consistency'] = (soft_token /self.head_num) * self.soft_token_consistency_weight
        end_points['object_query_consistency_loss'] = (object_query / self.head_num) * self.object_query_consistency_weight
        end_points['text_token_consistency_loss'] = (text_token / self.head_num) * self.text_token_consistency_weight

        end_points['consistency_loss'] =  end_points['box_consistency_loss'] + end_points['box_giou_consistency_loss'] +\
            end_points['soft_token_consistency'] + end_points['object_query_consistency_loss'] + end_points['text_token_consistency_loss']
        

        return end_points

        

      
    def compute_kl_div(self,token1,token2):
        
        #todo does reconstruct student necessary 

        return F.kl_div(F.log_softmax(token1, dim=2) , F.softmax(token2, dim=2) , reduction='mean')




    def  get_loss(self,student_out, teacher_out):
        
        
        #* 0. assume there are len(mask) target
        #*1. end_points['pred_logits'] X ema_end_points['pred_logits'][mask]  as the class cost or  soft token loss ,
        #*2. end_points['pred_boxes'] X ema_end_points[“pred_boxes”][mask] as box distance/GIoU losses

        mask= teacher_out['pred_sem_cls'] != 255
        target = [
                    {"boxes" :teacher_out['pred_boxes'][idx][m] ,"positive_map":teacher_out['pred_logits'][idx][m]} 
                    for idx, m in enumerate(mask)
                ]
        
        matcher = ConsistencyHungarianMatcher(cost_class=1,cost_bbox=5,cost_giou=2,soft_token=True)
        indices = matcher(student_out,target)

        num_boxes = sum(len(inds[1]) for inds in indices)#* num_boxes represent how many boxes target has.
        num_boxes = torch.as_tensor(
                [num_boxes], dtype=torch.float,
                device=next(iter(student_out.values())).device
            )

        self.num_boxes = num_boxes

        if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)

        
        center_size_consistency_loss,giou_consistency_loss = self.get_box_consistency(indices,student_out,target)

        soft_token_consistency = self.get_soft_token_consistency(indices,student_out,target)

        return center_size_consistency_loss,giou_consistency_loss,soft_token_consistency

        




    

    def get_soft_token_consistency(self,indices,student_out,target):

        logits = student_out["pred_logits"].log_softmax(-1)  # (B, Q, 256)
        

        # todo make target['positive_map'] to be  [0,1]
        positive_map = torch.cat([t["positive_map"] for t in target])

        # Trick to get target indices across batches
        src_idx = _get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(target[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        #? Labels, by default lines map to the last element, no_object
        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1 #* 匹配到最后一个表示no matched
        # target_sim[src_idx] = tgt_pos
        target_sim[src_idx] = F.sigmoid(tgt_pos)
        
        
        # embed()
        entropy = torch.log(target_sim + 1e-6) * target_sim
        # embed()
        loss_ce = (entropy - logits * target_sim).sum(-1)

        #!=====================================
        loss_ce = loss_ce.sum(-1)
        loss_ce = (loss_ce<torch.quantile(loss_ce, self.EMA_CLIP )) * loss_ce
        #!=====================================

        loss_ce = loss_ce.sum() / self.num_boxes
        
        return loss_ce

        
    def get_box_consistency(self,indices,student_out,target):
        
        #* for boxes 
        idx = _get_src_permutation_idx(indices)
        src_boxes = student_out['pred_boxes'][idx]
        #* suppose there are many boxes in  each target sample  
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(target, indices) ], dim=0)
        center_size_consistency_loss = (
                F.l1_loss(
                    src_boxes[..., :3], target_boxes[..., :3],
                    reduction='none'
                )
                + 0.2 * F.l1_loss(
                    src_boxes[..., 3:], target_boxes[..., 3:],
                    reduction='none'
                )
            )

        #* EMA CLIP
        center_size_consistency_loss = center_size_consistency_loss.sum(-1)
        center_size_consistency_loss = (center_size_consistency_loss<torch.quantile(center_size_consistency_loss, self.EMA_CLIP )) * center_size_consistency_loss


        center_size_consistency_loss= center_size_consistency_loss.sum() / self.num_boxes 
        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))        

        
        loss_giou = (loss_giou<torch.quantile(loss_giou, self.EMA_CLIP )) * loss_giou

        giou_consistency_loss= loss_giou.sum() / self.num_boxes

        return center_size_consistency_loss,giou_consistency_loss








'''
description:  获取 model 输出的 对应的目标, 也就是预测的span 取最大值后不是匹配到255的目标
param {*} end_points
param {*} prefix
return {*}
'''
def parse_endpoint(end_points,prefix):
    #* end_points['last_sem_cls_scores']  : [B,query_num,distribution for tokens(256)]  
    #* 1. 对 分布取softmax 
    #* 2. 取最大值的index, 如果最大值的索引是255 则 需要 
    #* 3. 对 分布取softmax 

    output = {}
    if 'proj_tokens' in end_points:
        output['proj_tokens'] = end_points['proj_tokens'] #!!!!!!!!
        output['proj_queries'] = end_points[f'{prefix}proj_queries']
        output['tokenized'] = end_points['tokenized']

    #* Get predicted boxes and labels, why the K equals to 256,n_class equals to 256, Q equals to 256
    pred_center = end_points[f'{prefix}center']  # B, K, 3
    pred_size = end_points[f'{prefix}pred_size']  # (B,K,3) (l,w,h)
    output["pred_boxes"] = torch.cat([pred_center, pred_size], dim=-1)
     

    output['pred_logits'] = end_points[f'{prefix}sem_cls_scores'] #* the soft token span logit, [B,query_number,token_span_range(256)]

    output["pred_sem_cls"] = torch.argmax(torch.nn.functional.softmax(output['pred_logits'],dim=-1) [..., :], -1)

    
    
    return output





'''
description:  对detector 的bbox进行数据增强 ,#! 没有加入噪声
param {*} self
param {*} bboxes
param {*} augmentations
return {*}
'''
def transformation_box(bboxes,augmentations):
    if len(augmentations) == 0:
        return 

        
    B,N,_=bboxes.shape

    all_det_pts = box2points(bboxes).view(B,-1, 3)

    for idx, tmp in enumerate(augmentations['yz_flip']): 
        if tmp:
            all_det_pts[idx,:, 0] = -all_det_pts[idx,:, 0]


    for idx, tmp in enumerate(augmentations['xz_flip']): 
        if tmp:
            all_det_pts[idx,:, 1] = -all_det_pts[idx,:, 1]

            
    all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
    all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
    all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])

  


    B2,N2,_=all_det_pts.shape
    all_det_pts += augmentations['shift'].cuda()
    all_det_pts *= augmentations['scale'].view(B,1,1).repeat(1,N2,3).cuda()

    bboxes = points2box(all_det_pts.reshape(B,-1, 8, 3))
    return bboxes


    