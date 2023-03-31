

import json
import os

import os.path as osp
import shutil
import numpy as np

import torch 
import torch.nn as nn
from loguru import logger 

from my_utils.pc_utils import write_pc_as_ply

import argparse
from collections import OrderedDict


from glob import glob

import os.path as osp


import shutil
from tqdm import tqdm
import os
import multiprocessing as mp


from IPython import embed


def parse_semi_supervise_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=8, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')



    # Data
    parser.add_argument('--batch_size', type=str, default="2,8",
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['sr3d'],
                        nargs='+', help='list of datasets to train on')
    

    parser.add_argument('--test_dataset', default='sr3d')
    #!+=======
    parser.add_argument('--unlabel-dataset-root', default=None)
    #!+=======
    parser.add_argument('--data_root', default='datasets/')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-3, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=10, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='logs/bdetr',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=1)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=1)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int,default=-1,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)
    parser.add_argument('--reduce_lr', action='store_true')

    #* mine args 
    #* semi supervise 
    parser.add_argument('--box_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--box_giou_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--soft_token_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--object_query_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--text_token_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--rampup_length', type=float, default=None, help='rampup_length')
    parser.add_argument('--labeled_ratio', default=None, type=float,help=' labeled datasets ratio ')
    
    #* others 
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument('--vis-save-path', default=None, type=str)
    parser.add_argument('--upload-wandb',action='store_true', help="upload to wandb or not ?")
    parser.add_argument('--save-input-output',action='store_true', help="save-input-output")
    parser.add_argument('--use-tkps',action='store_true', help="use-tkps")
    parser.add_argument('--lr_decay_intermediate',action='store_true')


    parser.add_argument('--ema-decay', default=None, type=float,help=' EMA decay parameter ')
    parser.add_argument('--ema-decay-after-rampup', default=None, type=float,help=' EMA decay parameter ')
    parser.add_argument('--ema-full-supervise', action='store_true',help='ema-full-supervise ')

    


    args, _ = parser.parse_known_args()
    args.eval = args.eval or args.eval_train



    args.use_color = True
    args.use_soft_token_loss=True
    args.use_contrastive_align=True
    args.self_attend=True
    
    # --use_color
    # --use_soft_token_loss
    # --use_contrastive_align
    # --self_attend
    # --use-tkps

    if args.labeled_ratio is not None :
        print(f"origin decay epoch : {args.lr_decay_epochs},opt.labeled_ratio : {args.labeled_ratio}")
        args.lr_decay_epochs  = (np.array(args.lr_decay_epochs) //  args.labeled_ratio).astype(np.int64).tolist()
        print(f"after calibration, decay epoch : {args.lr_decay_epochs}")

    return args



def parse_option():

    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=8, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    
    parser.add_argument('--dataset', type=str, default=['sr3d'],
                        nargs='+', help='list of datasets to train on')
    
    parser.add_argument('--test_dataset', default='sr3d')
    parser.add_argument('--data_root', default='datasets/')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true')
    
    parser.add_argument('--butd', action='store_true')
    
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-3, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='logs/bdetr',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=1)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=1)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int,default=-1,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-scanrefer', default=False, action='store_true',help=' generate the pred.json for the ')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)
    parser.add_argument('--reduce_lr', action='store_true')

    #* mine args 
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument('--vis-save-path', default=None, type=str)
    parser.add_argument('--upload-wandb',action='store_true', help="upload to wandb or not ?")
    parser.add_argument('--labeled_ratio', default=None, type=float,help=' labeled datasets ratio ')
    parser.add_argument('--use-tkps',action='store_true', help="use-tkps")
    parser.add_argument('--lr_decay_intermediate',action='store_true')

    args, _ = parser.parse_known_args()
    args.eval = args.eval or args.eval_train


    args.use_color = True
    args.use_soft_token_loss=True
    args.use_contrastive_align=True
    args.self_attend=True


    if args.labeled_ratio is not None :
        print(f"origin decay epoch : {args.lr_decay_epochs},opt.labeled_ratio : {args.labeled_ratio}")
        args.lr_decay_epochs  = (np.array(args.lr_decay_epochs) //  args.labeled_ratio).astype(np.int64).tolist()
        print(f"after calibration, decay epoch : {args.lr_decay_epochs}")



    return args




'''
description:  将分布式存储的模型转正常model
param {*} model
return {*}
'''
def detach_module(model):
    if len(list(model.keys())[0].split('.')) <=6:
        return model

    
    

    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
        new_state_dict[name] = v #新字典的key值对应的value一一对应

    return new_state_dict 


def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0


    #todo checkpoint['model']  delete ".module"
    # if distributed2common:
    
    
    common_model = detach_module(checkpoint['model'])
    # list(common_model.keys())[0]
    model.load_state_dict(common_model, True)
    # else :
    #     model.load_state_dict(checkpoint['model'], True)
    


    if not args.eval and not args.reduce_lr:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()




def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False,is_best=False,prefix=None):
    """Save checkpoint if requested."""
    spath = None
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        
        
        if is_best:
            if prefix is not None :
                spath = os.path.join(args.log_dir, f'{prefix}ckpt_epoch_{epoch}_best.pth')
            else :
                spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}_best.pth')
        else :
            spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')

        
            

        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")
    return spath




def make_dirs(path):

    if  not osp.exists(path):
        os.makedirs(path)
        


        
'''
description:  存储eval的结果
param {*} save_dir
param {*} epoch
param {*} performance
param {*} best_performce
return {*}
'''
def save_res(save_dir,epoch,performance,best_performce):
    is_best=False
    

    acc_key = list(performance.keys())[0]
    if performance is not None and performance[acc_key] > best_performce:
        is_best=True

    with open(save_dir, 'a+') as f :
        f.write( f"epoch:{epoch},"+','.join(["%s:%.4f"%(k,v) for k,v in performance.items()])+f" {is_best} \n")
    return is_best,performance[acc_key]





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
    

    


def move_dir_file(src_dir,target_dir):

    
    for f in os.listdir(src_dir):
        shutil.move(osp.join(src_dir,f),osp.join(target_dir,f))
        



def dump_json(path,json_data):

    with open (path , 'w') as f :
        
        json.dump(json_data,f)


def load_json(path):

    with open (path , 'r') as f :
        return json.load(f)



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
def read_refer_it_3D_txt(path = "~/exp/butd_detr/data/meta_data/sr3d_test_scans.txt"):
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
            
   



'''
description:  存储model 预测的结果用于上传ScanRefer evalution server , 
param {*} end_points : 输入一个batch 的 forward results, 返回  一个batch 的parse 结果
return {*}
'''
def my_parse_prediction(end_points):
    prefix="last_"
    query_dist_map = end_points[f'{prefix}sem_cls_scores'].softmax(-1) #* 
    objectness_preds_batch = torch.argmax(query_dist_map, 2).long() #* 等于255 应该是没有匹配到文本token的, 如paper解释的一样
    pred_masks = (objectness_preds_batch !=255).float()
    
    for i in range(pred_masks.shape[0]):
        # compute the iou 
        #* 存在一个utterence 有多个匹配的情况!!!  不知道选哪个?   choose first one for now 
        if pred_masks[i].sum() !=0 :

            matched_obj_size =end_points[f'{prefix}pred_size'][i][pred_masks[i]==1][0].detach().cpu().numpy()
            matched_obj_center =end_points[f'{prefix}center'][i][pred_masks[i]==1][0].detach().cpu().numpy() 
            matched_obj_xyz =end_points[f'{prefix}base_xyz'][i][pred_masks[i]==1][0].detach().cpu().numpy() 


            _bbox = get_3d_box(matched_obj_size,0, matched_obj_center) #* angle 不知道
            
            pred_data = {
                "scene_id": end_points["scan_ids"][i],
                "object_id": end_points["target_id"][i],
                "ann_id": end_points['ann_id'][i],
                "bbox": _bbox.tolist(),
                "unique_multiple":  end_points["is_unique"][i].item()==False, #* return true means multiple 
                "others": 1 if end_points["target_cid"][i] == 17 else 0
            }
            pred_bboxes.append(pred_data)
    return pred_bboxes



def save_box(box,path):
    np.savetxt(path,box,fmt='%s')

def save_pc(pc,path):
    write_pc_as_ply(pc,path)


'''
description: 
param {*} data
param {*} scene_name
param {*} save_root
param {*} has_color
param {*} flag
return {*}
'''
def save_for_vis(box,pc,save_path,scene_name,bidx,flag = "debug",idx=0,save = True):
      #* for teacher or student 
    if save :
        save_pc(pc,os.path.join(save_path, '%s_%d_%s_gt_%s.ply'%(scene_name,idx,bidx,flag)))
    
    save_box(box,os.path.join(save_path,'%s_%d_%s_box_%s.txt'%(scene_name,idx,bidx,flag)) )





def save_txt(data,path):
    

    with open('logs/debug/final_list.txt','w')as f :
        f.write(data)


    with open(path,'w')as f :
        f.write(data)





'''
description:  for huanang detector 
param {*} datasets
return {*}
'''
def save_pc_for_detector(datasets):

    save_dir = "datasets/scannet_pc"
    make_dirs(save_dir)

    length = datasets.__len__()
    print(f"len : {length}")
    for idx in range(length):
        pc,scane_name = datasets.get_origin_data(idx)
        save_path = osp.join(save_dir,f"{scane_name}_pc.npy")
        np.save(save_path,pc)




def remove_file(old_path):
    
    if old_path is not None:
        os.remove(old_path)





def copy_ARKitScens():
    
    src_path = "~/exp/butd_detr/datasets/arkitscenes/dataset/3dod"
    tgt_path='~/exp/butd_detr/datasets/ARKitScenes/dataset/3dod'

    splits = {
        'train':"Training",'valid':"Validation"
    }


    for split,item in splits.items():
        src_all_sample_pathes  = os.listdir(osp.join(src_path,item))

        for p in tqdm(src_all_sample_pathes):
            src = osp.join(src_path,item,p,f'{p}_offline_prepared_data_2')
            tgt = osp.join(tgt_path,item,p,f'{p}_offline_prepared_data_2')

            if osp.exists(tgt):
                shutil.rmtree(tgt)
                
            

            if osp.exists(src):
                shutil.copytree(src,tgt)
            





def delete_ckpt_except_last_one(path):
    for o in os.listdir(path ):
        all_ckpt = glob(osp.join(path,o)+"/*.pth")
        num = len(all_ckpt)
        if num >1:
            all_ckpt = sorted(all_ckpt,key = lambda x : int(x.split('/')[-1].split('_')[-2]))
            
            for ckpt in all_ckpt[:-1]:
                os.remove(ckpt)
                print(ckpt,"has deleted")
            print("=======================================")








if __name__ == "__main__":
    # read_refer_it_3D_txt()
    
    
    #* 生成labeled datasets for SR3D
    for x in np.linspace(0.1,0.9,9):
        generate_SR3D_labeled_scene_txt(round(x,1))


    #* demo for save_pc_for_detector
    # save_pc_for_detector(train_dataset)
    # save_pc_for_detector(val_dataset)
    # pc,scane_name=val_dataset.get_origin_data(0)


    #* delete ckpy try
    # delete_ckpt_except_last_one(path)

