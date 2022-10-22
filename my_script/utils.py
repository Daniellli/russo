'''
Author: xushaocong
Date: 2022-10-02 20:04:19
LastEditTime: 2022-10-22 11:41:22
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/my_script/utils.py
email: xushaocong@stu.xmu.edu.cn
'''


import os

import os.path as osp
import numpy as np

import torch 
import torch.nn as nn
from loguru import logger 

def make_dirs(path):

    if  not osp.exists(path):
        os.makedirs(path)
        



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
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1,-1,M,-1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1,N,-1,-1)
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
description:  read txt file 
param {*} path
return {*}
'''
def readtxt(path):
    data = None
    with open(path,'r') as f :
        data = f.read()
    return data




'''
description:  read the meta data of SR3D 
param {*} path : val == test  in SR3D 
return {*}
'''
def read_refer_it_3D_txt(path = "/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/sr3d_test_scans.txt"):
    data  = readtxt(path)
    data = data[1:-1].split(',')
    data = [x.replace('"',"").strip() for x in data]
    logger.info(f"scene number : {len(data)}")
    return data




def generate_SR3D_labeled_scene_txt(labeled_ratio):
    
    split='train'

    with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
        scan_ids = set(eval(f.read()))

    num_scans = len(scan_ids)
    logger.info(f"read {num_scans} scenes ") 
    num_labeled_scans = int(num_scans*labeled_ratio)

    choices = np.random.choice(num_scans, num_labeled_scans, replace=False)#* 从num_scans 挑选num_labeled_scans 个场景 出来 
    labeled_scan_names = list(np.array(list(scan_ids))[choices])
    
    with open(os.path.join('data/meta_data/sr3d_{}_{}.txt'.format(split,labeled_ratio)), 'w') as f:
        f.write('\n'.join(labeled_scan_names))
    
    logger.info('\tSelected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names),len(scan_ids )- len(labeled_scan_names)))

    


'''
description:  打印字典所有的key
param {*} data
return {*}
'''
def get_attr(data):
    return list(data.keys())

'''
description: 打印字典每个数据的shape
param {*} data
return {*}
'''
def print_attr_shape(data):
    for k,v in data.items():
        if   isinstance(v,str):
            
            logger.info(f"{k} : {v}")
        elif   isinstance(v,bool):
            logger.info(f"{k} : {v}")
        elif   isinstance(v,int):
            logger.info(f"{k} : {v}")
        elif   isinstance(v,dict):
            logger.info(f"printing {k} \n==========================================================================")

            for kk,vv in v.items():
                if   isinstance(vv,bool):
                    logger.info(f"{kk} : {vv}")
                elif   isinstance(vv,float):
                    logger.info(f"{kk} : {vv}")
                else :
                    logger.info(f"{kk} : {vv.shape}")
            logger.info(f"\n==========================================================================")
                
        elif v is not None:
            logger.info(f"{k} : {v.shape}")
        else :
            logger.info(f"{k} some thing wrong ")
            
   


if __name__ == "__main__":
    # read_refer_it_3D_txt()
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/sr3d_train_scans.txt")
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/nr3d_test_scans.txt")
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/nr3d_train_scans.txt")
    
    #* 生成labeled datasets for SR3D
    for x in np.linspace(0.1,0.9,9):
        generate_SR3D_labeled_scene_txt(round(x,1))

