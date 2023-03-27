
 



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



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


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
description: 
param {*} bbox
param {*} ema_bbox
param {*} mask
param {*} logit : the  object logit 
return {*}
'''
def compute_bbox_center_consistency_loss(center, ema_center,mask=None,logit=None):
   
    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #* ind1 (B, num_proposal): find the ema_center index closest to center
    
    # 
    #TODO: use both dist1 and dist2 or only use dist1
    if mask is not None :

        # dist2 = (dist2<torch.quantile(dist2, 0.85)) * dist2

        return dist2[mask].sum(),ind2
        #* 返回 loss,    teacher center 向student  center 对齐的索引

        # return (dist.sum(-1)/(mask.sum(-1)+1e-10)).sum(),ind2
    else :
        # return (dist1+dist2).mean(),ind2
        # dist2 = logit*dist2 #* the obj probility to filter 

        dist_ = (dist2<torch.quantile(dist2, 0.25)) * dist2
        return dist_.mean(),ind2







'''
description:  compute the loss between two  token map 
param {*} cls_scores
param {*} ema_cls_scores
param {*} map_ind
param {*} mask
return {*}
'''
def compute_token_map_consistency_loss(cls_scores, ema_cls_scores,map_ind,mask=None):
    #* ? 
    B,Q,T=cls_scores.shape
    cls_log_prob = F.log_softmax(cls_scores, dim=2) #(B, num_proposal, num_class)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)

    # todo : 
    cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])#* 根据map_ind 重新组织cls_log_prob (student out)
    if mask is not None:    
        
        # class_consistency_loss = F.kl_div(cls_log_prob_aligned[mask], ema_cls_prob[mask], reduction='none')
        # class_consistency_loss[~mask] =0
        #* class_consistency_loss : [B,Q,T] 
        #todo does  it need to multiple by 2 according to  SESS? 
        # return (class_consistency_loss.mean(-1).sum(-1)/(mask.sum(-1)+1e-10)).sum()/B

        return F.kl_div(cls_log_prob_aligned[mask], ema_cls_prob[mask], reduction='mean')
    else :
        class_consistency_loss = F.kl_div(cls_log_prob_aligned, ema_cls_prob, reduction='none')
        return class_consistency_loss.mean()*2



    


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


    


'''
description: 
param {*} end_points
param {*} ema_end_points
param {*} map_ind
param {*} config
return {*}
'''
def compute_size_consistency_loss(size, ema_size, map_ind, mask):
    
    B,N,_=size.shape
    size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])


    if mask is not None:    
        dist2= F.mse_loss(size_aligned, ema_size,reduction='none')

        # dist2 = (dist2<torch.quantile(dist2, 0.85)) * dist2
        # return dist2[mask].mean()

        return dist2[mask].sum()

        

    else :
        size_consistency_loss = F.mse_loss(size_aligned, ema_size,reduction='none')
        return size_consistency_loss.mean()

    

def _get_src_permutation_idx( indices):
    # permute predictions following indices
    batch_idx = torch.cat([
        torch.full_like(src, i) for i, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx 

def _get_tgt_permutation_idx( indices):
    # permute targets following indices
    batch_idx = torch.cat([
        torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
    ])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx




    
'''
description:   
param {*} end_points
param {*} ema_end_points : teacher
param {*} prefix
return {*}
'''
def compute_refer_consistency_loss(end_points, ema_end_points,augmentation, prefix="last_"):
    
    student_out=parse_endpoint(end_points,prefix)
    teacher_out=parse_endpoint(ema_end_points,prefix)
    
    # processong augmentaiton for 
    if augmentation is not None and len(augmentation.keys()) >0:
        teacher_out['pred_boxes'] = transformation_box(teacher_out['pred_boxes'],augmentation)
    
    

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


    if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)


    #todo compute consistency loss by the hugariun matching
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
    center_size_consistency_loss= center_size_consistency_loss.sum() / num_boxes 
    

    loss_giou = 1 - torch.diag(generalized_box_iou3d(
        box_cxcyczwhd_to_xyzxyz(src_boxes),
        box_cxcyczwhd_to_xyzxyz(target_boxes)))        
    giou_consistency_loss= loss_giou.sum() / num_boxes


    #* for soft token consistency loss


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
    target_sim[src_idx] = tgt_pos

    entropy = torch.log(target_sim + 1e-6) * target_sim
    loss_ce = (entropy - logits * target_sim).sum(-1)
    loss_ce = loss_ce.sum() / num_boxes





    query_contrastive_consistency_loss = 1 - F.cosine_similarity(student_out['proj_queries'], teacher_out['proj_queries'])

    token_contrastive_consistency_loss = 1 - F.cosine_similarity(student_out['proj_tokens'], teacher_out['proj_tokens'])



    # center_loss,teacher2student_map_idx = compute_bbox_center_consistency_loss(student_out['pred_boxes'][:,:,:3],teacher_out['pred_boxes'][:,:,:3],mask)
    # size_loss = compute_size_consistency_loss(student_out['pred_boxes'][:,:,3:],teacher_out['pred_boxes'][:,:,3:],teacher2student_map_idx,mask)

    # soft_token_loss=compute_token_map_consistency_loss(student_out['pred_logits'],teacher_out['pred_logits'],teacher2student_map_idx,mask= mask)

    # query_consistent_loss=compute_query_consistency_loss(student_out['proj_queries'],teacher_out['proj_queries'],teacher2student_map_idx)
    
    # text_consistent_loss=compute_text_consistency_loss(student_out['proj_tokens'],teacher_out['proj_tokens'],teacher2student_map_idx)


    return center_loss,soft_token_loss,size_loss,query_consistent_loss,text_consistent_loss
    



'''
description: 计算queries 之间的 距离
param {*} student_querys
param {*} teacher_query
param {*} map_idx
return {*}
'''
def compute_query_consistency_loss(student_query,teacher_query,map_idx):
       
    __student_log_query = F.log_softmax(student_query, dim=2) #(B, num_proposal, num_class)
    __teacher_query = F.softmax(teacher_query, dim=2) #(B, num_proposal, num_class)

    student_query_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(__student_log_query, map_idx)])#* 根据map_ind 重新组织cls_log_prob (student out)
    return F.kl_div(student_query_aligned, __teacher_query, reduction='mean')



'''
description: 计算queries 之间的 距离
param {*} student_query
param {*} teacher_query
param {*} map_idx
return {*}
'''
def compute_text_consistency_loss(student_text,teacher_text,map_idx):

    return F.kl_div( F.log_softmax(student_text, dim=2) , F.softmax(teacher_text, dim=2), reduction='mean')*2



def get_consistency_loss(end_points, ema_end_points,augmentation):
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
    
    soft_token_consistency_loss_sum = torch.tensor(0.).cuda()
    center_consistency_loss_sum = torch.tensor(0.).cuda()
    size_consistency_loss_sum = torch.tensor(0.).cuda()
    
    query_consistency_loss_sum = torch.tensor(0.).cuda()
    text_consistency_loss_sum = torch.tensor(0.).cuda()

    
    prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(5)] #* 6 heads + proposal 
    
    
    DEBUG = False
    if DEBUG:
        make_dirs(DEBUG_FILT)
        check_transformation(end_points, prefixes)
        check_transformation(ema_end_points, prefixes,is_student=False,pc_param_name='point_clouds')
        check_teacher_box_aug_back(ema_end_points,augmentation,prefixes)
        


    for prefix in prefixes:

        center_loss,soft_token_loss,size_loss,query_consistent_loss,text_consistent_loss = compute_refer_consistency_loss(end_points, ema_end_points, augmentation,prefix=prefix)
        center_consistency_loss_sum+=center_loss
        soft_token_consistency_loss_sum+=soft_token_loss
        size_consistency_loss_sum+=size_loss
        query_consistency_loss_sum+=query_consistent_loss
        text_consistency_loss_sum+=text_consistent_loss
        
    end_points['soft_token_consistency_loss'] = soft_token_consistency_loss_sum / len(prefixes)
    end_points['center_consistency_loss'] = center_consistency_loss_sum / len(prefixes)
    end_points['size_consistency_loss'] = size_consistency_loss_sum / len(prefixes)

    end_points['query_consistency_loss'] = query_consistency_loss_sum / len(prefixes)
    end_points['text_consistency_loss'] = text_consistency_loss_sum / len(prefixes)


    return end_points














'''
description:  only for debug 
param {*} ema_end_points
param {*} augmentation
param {*} prefixes
return {*}
'''
def check_teacher_box_aug_back(ema_end_points,augmentation,prefixes):

    for p in prefixes:
        teacher_out=parse_endpoint(ema_end_points,p)
        teacher_boxes=teacher_out['pred_boxes'].clone()
        
        teacher_boxes=  transformation_box(teacher_boxes,augmentation)

        scene_ids =ema_end_points['scan_ids']
        flag = "teacher"

        for idx in range(len(scene_ids)):

            mask_len = (teacher_out['tokenized']['attention_mask']==1).sum(-1) #* 获取每个description 经过language model 后的token  set 实际长度
            activated_map = get_activated_map(teacher_out['pred_logits'])
            mask =torch.cat([(activated_map[idx]<desc_len).unsqueeze(0) for idx,desc_len in enumerate(mask_len)],dim=0)#* calculate the mask 

            #* open3d  format bounding box 
            np.savetxt(osp.join(DEBUG_FILT,scene_ids[idx],'%s_box_%s_%s_trans.txt'%(scene_ids[idx],flag,p)),teacher_boxes[idx][mask[idx]].clone().detach().cpu().numpy(),fmt='%s')
            



'''
description:  保存 pc 和bbox, 用于可视化进行debug
param {*} end_points
param {*} prefixes
param {*} is_student : 是否是student model,默认是
param {*} pc_param_name :  改bbox对应的点云参数名字 ,默认student 的 point_clouds , 也就是增强后的点云
return {*}
'''
def check_transformation(end_points,prefixes,is_student = True,pc_param_name='pc_before_aug'):
    scene_ids = end_points['scan_ids']
    
    

    for s in scene_ids:
        make_dirs(osp.join(DEBUG_FILT,s))

    
    FLAG = "student" if is_student else 'teacher'
    
    student_pc = end_points[pc_param_name] #* teacher  pc is different with student's

    
    #* save pc 
    for idx in range(len(scene_ids)):
        write_pc_as_ply(
                    student_pc[idx].clone().detach().cpu().numpy(),
                    os.path.join(DEBUG_FILT, scene_ids[idx],'%s_gt_%s.ply'%(scene_ids[idx],FLAG))
                )


    for prefix in prefixes:
        student_out  = parse_endpoint(end_points,prefix)
        bbox  = student_out['pred_boxes']
        for idx in range(len(scene_ids)):

            mask_len = (student_out['tokenized']['attention_mask']==1).sum(-1) #* 获取每个description 经过language model 后的token  set 实际长度
            activated_map = get_activated_map(student_out['pred_logits'])
            mask =torch.cat([(activated_map[idx]<desc_len).unsqueeze(0) for idx,desc_len in enumerate(mask_len)],dim=0)#* calculate the mask 

            #* open3d  format bounding box 
            np.savetxt(osp.join(DEBUG_FILT,scene_ids[idx],'%s_box_%s_%s.txt'%(scene_ids[idx],FLAG,prefix)),bbox[idx][mask[idx]].clone().detach().cpu().numpy(),fmt='%s')
            





    

'''
description:  给定一个model 推理输出的 [B,query_num,token_map_size] 的 embedding, 对token_map_size 取响应最大的token position得[B,query_num] 
param {*} teacher_logit
return {*}
'''
def get_activated_map(teacher_logit):#todo:  默认每个bbox 只响应一个 token ; how to optimize 
    query_dist_map =teacher_logit.softmax(-1) #* softmax 找到对应的token,  每个channel对应的一个token , 共256个token 
    target = torch.argmax(query_dist_map, 2).long() #* 等于255 应该是没有匹配到文本token的, 如paper解释的一样

    return target

  
