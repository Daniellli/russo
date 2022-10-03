'''
Author: xushaocong
Date: 2022-09-22 23:13:23
LastEditTime: 2022-10-03 16:24:47
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/my_script/consistant_loss.py
email: xushaocong@stu.xmu.edu.cn
'''




 



import sys

sys.path.append("/data/xusc/exp/butd_detr")
import torch
import torch.nn as nn


import os.path as osp

from my_script.pc_utils import *



import torch.nn.functional as F


DEBUG_FILT = "/data/xusc/exp/butd_detr/logs/debug"



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
    pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
    pred_logits = end_points[f'{prefix}sem_cls_scores']  # (B, Q, n_class)
    output['pred_logits'] = pred_logits
    output["pred_boxes"] = pred_bbox

    return output




'''
description:  Normalized distances of points and gt centroids,
param {*} student_bbox
param {*} teacher_bbox
return {*}
'''
def  bbox_distance_loss(student_bbox,teacher_bbox,mask=None):
    # B,C,_=student_bbox.shape #todo dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)
    student_bbox_center = student_bbox[:,:,:3]
    gt_center = teacher_bbox[:,:,:3]
    gt_size = teacher_bbox[:,:,3:]

    delta_xyz = student_bbox_center - gt_center
    
    delta_xyz = delta_xyz / (gt_size+ 1e-6)  # (B, K, G, 3)
    
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    

    if mask is not None:
        #* with regularization 
        euclidean_dist1[~mask] = 0

    return euclidean_dist1.sum()

    

'''
description:  给定一个model 推理输出的 [B,query_num,token_map_size] 的 embedding, 对token_map_size 取响应最大的token position得[B,query_num] 
param {*} teacher_logit
return {*}
'''
def get_activated_map(teacher_logit):#todo:  默认每个bbox 只响应一个 token ; how to optimize 
    query_dist_map =teacher_logit.softmax(-1) #* softmax 找到对应的token,  每个channel对应的一个token , 共256个token 
    target = torch.argmax(query_dist_map, 2).long() #* 等于255 应该是没有匹配到文本token的, 如paper解释的一样

    return target


def soft_token_consist_loss(student_logit,teacher_logit,mask = None):
    teacher = get_activated_map(teacher_logit) 
    # B,query_num,token_span=student_logit.shape
    if mask is not None:
        loss = F.cross_entropy(student_logit, teacher.long(),weight=None,reduction='none')
        loss[~mask] = 0
        loss = loss.sum()
    else :
        loss = F.cross_entropy(student_logit, teacher.long(),weight=None,reduction='sum')

    return loss
  


'''
description:  根据focal loss的公式编写
param {*} self
param {*} logit
param {*} target
param {*} gamma : focal parameter
param {*} alpha 样本均衡参数
return {*}
'''
def focalLoss(logit, target, gamma=2, alpha=0.5,weight=None):
    B, C, map_size = logit.size()

    criterion = nn.CrossEntropyLoss(weight=weight,reduction='sum')

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    logpt = -criterion(logit, target.long())

    pt = torch.exp(logpt)  
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    loss /= B
    return loss

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2




'''
description: 
param {*} bbox
param {*} ema_bbox
param {*} mask
return {*}
'''
def compute_bbox_consistency_loss(bbox, ema_bbox,mask=None):
   
    center = bbox[:,:,:3]
    size = bbox[:,:,3:]

    ema_center = ema_bbox[:,:,:3]
    ema_size = ema_bbox[:,:,3:]

    B,query_num,_=bbox.shape
    if mask is not None : 
        #*这样就不能clip 了
        center[~mask]=  1e+6
        ema_center[~mask] =1e+6

    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #* ind1 (B, num_proposal): find the ema_center index closest to center
    

    # rearrange_center=torch.cat([center[b_idx,ind1[b_idx],:].unsqueeze(0) for b_idx in range(B)],dim=0)
    # rearrange_ema_center=torch.cat([ema_center[b_idx,ind2[b_idx],:].unsqueeze(0) for b_idx in range(B)],dim=0)
    
    # (((rearrange_center - ema_center )**2).sum(dim=-1) == dist1).int().sum()
    # (((rearrange_ema_center - center )**2).sum(dim=-1) == dist2).int().sum()

    # rearrange_size =torch.cat([size[b_idx,ind1[b_idx],:].unsqueeze(0) for b_idx in range(B)],dim=0)
    # size_loss=F.mse_loss(rearrange_size, ema_size)
    #!=========================
    # border_line = torch.quantile(dist1,0.90)
    # dist1 = (dist1<border_line) * dist1
    # if mask is not None : 
    #     dist1[~mask] = 0
    #!=========================

    # return dist1.sum()
    # return (dist1.sum(-1)/mask.sum(-1)).sum()
    return (dist1.sum(-1)/(mask.sum(-1)+1e-10)).sum()



def compute_bbox_consistency_loss_size(bbox, ema_bbox,mask=None,clip=False):
   
    center = bbox[:,:,:3]
    size = bbox[:,:,3:]

    ema_center = ema_bbox[:,:,:3]
    ema_size = ema_bbox[:,:,3:]

    B,query_num,_=bbox.shape

    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #* ind1 (B, num_proposal): find the ema_center index closest to center
    
    rearrange_size =torch.cat([size[b_idx,ind1[b_idx],:].unsqueeze(0) for b_idx in range(B)],dim=0)
    # size_loss=((rearrange_size - ema_size)**2).mean() #* == F.mse_loss(rearrange_size, ema_size)
    
    size_loss=((rearrange_size - ema_size)**2)

    # if clip:
    #     border_line = torch.quantile(dist1,0.85)
    #     dist1 = (dist1<border_line) * dist1
    
    if mask is not None : 
        return size_loss[mask].sum()
        
    return size_loss.sum()

def soft_token_consist_loss_kl(cls_scores, ema_cls_scores,mask=None):
    #* ? 
    cls_log_prob = F.log_softmax(cls_scores, dim=2) #(B, num_proposal, num_class)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)

    # cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(cls_log_prob, ema_cls_prob, reduction='none')
    if mask is not None:    
        class_consistency_loss[~mask] =0
    

    return class_consistency_loss.sum()



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
    all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
    all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
    all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])

    for idx, tmp in enumerate(augmentations['yz_flip']): 
        if tmp:
            all_det_pts[idx,:, 0] = -all_det_pts[idx,:, 0]


    for idx, tmp in enumerate(augmentations['xz_flip']): 
        if tmp:
            all_det_pts[idx,:, 1] = -all_det_pts[idx,:, 1]


    B2,N2,_=all_det_pts.shape
    all_det_pts += augmentations['shift'].cuda()
    all_det_pts *= augmentations['scale'].view(B,1,1).repeat(1,N2,3).cuda()

    bboxes = points2box(all_det_pts.reshape(B,-1, 8, 3))
    return bboxes


    


def rot_x(pc, B_theta):
    """Rotate along x-axis."""
    B_theta = B_theta * np.pi / 180

    ans = []
    for theta in B_theta:
        tmp  = torch.stack([
            torch.tensor([torch.tensor(1.0), torch.tensor(0), torch.tensor(0)]),
            torch.stack([torch.tensor(0), torch.cos(theta), -torch.sin(theta)]),
            torch.stack([torch.tensor(0), torch.sin(theta), torch.cos(theta)])
        ],axis=0)
        ans.append(tmp)
    ans = torch.stack(ans).cuda()


    return torch.matmul(ans.float(),pc.permute(0,2,1).float()).permute(0,2,1)
 


def rot_y(pc, B_theta):
    """Rotate along y-axis."""
    B_theta = B_theta * np.pi / 180

    ans = []
    for theta in B_theta:
        tmp  = torch.stack([
            torch.stack([torch.cos(theta), torch.tensor(0), torch.sin(theta)]),
            torch.tensor([torch.tensor(0), torch.tensor(1.0), torch.tensor(0)]),
            torch.stack([-torch.sin(theta), torch.tensor(0), torch.cos(theta)])
        ],axis=0)
        ans.append(tmp)
    ans = torch.stack(ans).cuda()


    return torch.matmul(ans.float(),pc.permute(0,2,1).float()).permute(0,2,1)


def rot_z(pc, B_theta):
    """Rotate along z-axis."""
    
    B_theta = B_theta * np.pi / 180

    ans = [] 
    for theta in B_theta:
        tmp  = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta),torch.tensor(0)]),
            torch.stack([torch.sin(theta), torch.cos(theta),torch.tensor(0)]),
            torch.stack([torch.tensor(0),torch.tensor(0), torch.tensor(1)])
        ],axis=0)
        ans.append(tmp)
    ans = torch.stack(ans).cuda()

    return torch.matmul(ans.float(),pc.permute(0,2,1).float()).permute(0,2,1)
    





def box2points(box):
    B,N,_=box.shape
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:,:, :3] - (box[:,:, 3:] / 2)).transpose(2, 1).permute(1,0,2)
    x_max, y_max, z_max = (box[:,:, :3] + (box[:,:, 3:] / 2)).transpose(2, 1).permute(1,0,2)
    return torch.stack((
        torch.cat((x_min[:,:, None], y_min[:,:, None], z_min[:,:, None]), 2),
        torch.cat((x_min[:,:, None], y_max[:,:, None], z_min[:,:, None]), 2),
        torch.cat((x_max[:,:, None], y_min[:,:, None], z_min[:,:, None]), 2),
        torch.cat((x_max[:,:, None], y_max[:,:, None], z_min[:,:, None]), 2),
        torch.cat((x_min[:,:, None], y_min[:,:, None], z_max[:,:, None]), 2),
        torch.cat((x_min[:,:, None], y_max[:,:, None], z_max[:,:, None]), 2),
        torch.cat((x_max[:,:, None], y_min[:,:, None], z_max[:,:, None]), 2),
        torch.cat((x_max[:,:, None], y_max[:,:, None], z_max[:,:, None]), 2)
    ), axis=2)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    B,N,_,_=box.shape
    box = box.reshape(-1,8,3)
    return torch.cat((
        (box.min(1)[0] + box.max(1)[0]) / 2,
        box.max(1)[0] - box.min(1)[0]
    ), axis=1).reshape([B,N,6])
    

def make_dirs(path):

    if  not osp.exists(path):
        os.makedirs(path)
        

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
        teacher_bbox = transformation_box(teacher_out['pred_boxes'],augmentation)
    else :
        teacher_bbox = teacher_out['pred_boxes']


    #* 下面这行是成立的
    #* teacher_out['tokenized'] ['input_ids'] == student_out['tokenized'] ['input_ids']
    
    
    #*  只对token span 内的 pred target 计算loss
    mask_len = (teacher_out['tokenized']['attention_mask']==1).sum(-1) #* 获取每个description 经过language model 后的token  set 实际长度
    teacher_activated_token_map = get_activated_map(teacher_out['pred_logits'])
    mask =torch.cat([(teacher_activated_token_map[idx]<desc_len).unsqueeze(0) for idx,desc_len in enumerate(mask_len)],dim=0)#* calculate the mask 

    # bbox_loss = bbox_distance_loss(student_out['pred_boxes'],teacher_out['pred_boxes'],mask= mask)
    
    soft_token_loss = soft_token_consist_loss(student_out['pred_logits'],teacher_out['pred_logits'],mask= mask)
    # soft_token_kl_loss=soft_token_consist_loss_kl(student_out['pred_logits'],teacher_out['pred_logits'],mask= mask) #* 这个可能会比cross entropy更好
    

    # center_loss,size_loss = compute_bbox_consistency_loss(student_out['pred_boxes'],teacher_out['pred_boxes'])
    # center_loss = compute_bbox_consistency_loss(student_out['pred_boxes'],teacher_bbox,mask)
    # size_loss = compute_bbox_consistency_loss_size(student_out['pred_boxes'],teacher_bbox,mask)

    # student_out['proj_tokens']#* [B,token_num,64],  
    # student_out['proj_queries'] #*  [B,query_num,64]
    # student_out['tokenized']#* language model output ; this model is freezed ; [B,token_num]....

    
    # logger.info(f" soft_token_loss:{soft_token_loss}, soft_token_kl_loss : {soft_token_kl_loss}, center_loss:{center_loss},size_loss:{size_loss}")


    # return soft_token_loss,soft_token_kl_loss,center_loss,size_loss
    # return soft_token_loss
    return soft_token_loss
    






'''
description: 
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
    # soft_token_kl_consistency_loss_sum = torch.tensor(0.).cuda()
    # center_consistency_loss_sum = torch.tensor(0.).cuda()
    # size_consistency_loss_sum = torch.tensor(0.).cuda()

    
    prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(5)] #* 6 heads + proposal 
    # prefixes = ['last_']
    


    DEBUG = False
    if DEBUG:
        make_dirs(DEBUG_FILT)
        # end_points['pc_before_aug']
        # end_points['point_clouds']
        
        check_transformation(end_points, prefixes)
        check_transformation(ema_end_points, prefixes,is_student=False,pc_param_name='point_clouds')

        check_teacher_box_aug_back(ema_end_points,augmentation,prefixes)
        


    for prefix in prefixes:
        
        
        # soft_token_loss,soft_token_kl_loss,center_loss,size_loss= compute_refer_consistency_loss(end_points, ema_end_points, prefix=prefix)
        # center_loss,soft_token_kl_loss= compute_refer_consistency_loss(end_points, ema_end_points,augmentation, prefix=prefix)
        # center_loss= compute_refer_consistency_loss(end_points, ema_end_points,augmentation, prefix=prefix)
        soft_token_loss= compute_refer_consistency_loss(end_points, ema_end_points, augmentation,prefix=prefix)
        # center_loss= compute_refer_consistency_loss(end_points, ema_end_points, augmentation,prefix=prefix)
        # size_loss= compute_refer_consistency_loss(end_points, ema_end_points,augmentation, prefix=prefix)
        

        soft_token_consistency_loss_sum+=soft_token_loss
        # soft_token_kl_consistency_loss_sum+=soft_token_kl_loss

        # center_consistency_loss_sum+=center_loss
        # size_consistency_loss_sum+=size_loss
        
        
    end_points['soft_token_consistency_loss'] = soft_token_consistency_loss_sum / len(prefixes)
    # end_points['soft_token_kl_consistency_loss'] = soft_token_kl_consistency_loss_sum / len(prefixes)

    # end_points['center_consistency_loss'] = center_consistency_loss_sum / len(prefixes)
    # end_points['size_consistency_loss'] = size_consistency_loss_sum / len(prefixes)

    return end_points
