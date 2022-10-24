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
"""Shared utilities for all main scripts."""

import argparse
import json
import os
from posixpath import dirname
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger



# from my_script.vis_utils import *
from my_script.pc_utils import *

from loguru import logger




import os.path as osp

from IPython import embed

from collections import OrderedDict


#*=====================================
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)
#*=====================================

'''
description:  将分布式存储的模型转正常model
param {*} model
return {*}
'''
def detach_module(model):
    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
        new_state_dict[name] = v #新字典的key值对应的value一一对应

    return new_state_dict 


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
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
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
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true')
    #*========================
    parser.add_argument('--butd', action='store_true')
    #*========================
    parser.add_argument('--butd_gt', action='store_true')
    #*========================
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    #*========================
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
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
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int,
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
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument('--vis-save-path', default='', type=str)
    parser.add_argument('--upload-wandb',action='store_true', help="upload to wandb or not ?")
    parser.add_argument('--save-input-output',action='store_true', help="save-input-output")

    parser.add_argument('--scanrefer-test',action='store_true', help="scanrefer-test")


    parser.add_argument('--size_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--center_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--token_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')

    parser.add_argument('--query_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--text_consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')

    parser.add_argument('--labeled_ratio', default=0.2, type=float,help=' labeled datasets ratio ')
    parser.add_argument('--rampup_length', type=float, default=None, help='rampup_length')
    


    parser.add_argument('--use-tkps',action='store_true', help="use-tkps")

    parser.add_argument('--lr_decay_intermediate',action='store_true')



    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train

    return args


def load_checkpoint(args, model, optimizer, scheduler,distributed2common=False):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0


    #todo checkpoint['model']  delete ".module"
    if distributed2common:
        common_model = detach_module(checkpoint['model'])
        model.load_state_dict(common_model, True)
    else :
        model.load_state_dict(checkpoint['model'], True)
    


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

        


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]
        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )
        
        self.vis_save_path=osp.join(args.log_dir,'debug')
        os.makedirs(self.vis_save_path,exist_ok=True)
        
        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))
            

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        #* do not need to load train set when args.evals == True 
        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        #* 存在一个问题就是val 的数据抽取的不合法,在group_free_pred_bboxes_val 找不到对应的数据
        # Samplers and loaders
        # for k in train_dataset.__getitem__(1).keys():
        #     if hasattr(train_dataset.__getitem__(1)[k],"shape"):
        #         print(k,train_dataset.__getitem__(1)[k].shape)

        
        g = torch.Generator()
        g.manual_seed(0)
        
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        losses = ['boxes', 'labels']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        param_dicts = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone_net" not in n and "text_encoder" not in n
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone_net" in n and p.requires_grad
                ],
                "lr": args.lr_backbone
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr
            }
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer

    def main(self, args):

        
        #!======================= 避免数据跑到其他卡上
        torch.cuda.set_device(args.local_rank)
        logger.info(f"args.local_rank == {args.local_rank}")
        #!=======================
        
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        n_data = len(train_loader.dataset)
        self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)
   

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        scheduler = get_scheduler(optimizer, len(train_loader), args)
        

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda(args.local_rank)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )
        #!+===========mine
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cuda(args.local_rank))
        # model = DistributedDataParallel(model,device_ids=[args.local_rank],find_unused_parameters=True,broadcast_buffers = True) 
        #!+===========

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)

            #!=========================================
            #* 将milestone的第一个元素 也就是lr decay 提前到之后的第一个epoch
            if args.lr_decay_intermediate:    

                #* 用lr_decay_epochs 作为milestone
                n_iter_per_epoch=  len(train_loader)
                milestones={(m - args.warmup_epoch) * n_iter_per_epoch : 1  for m in args.lr_decay_epochs}
                scheduler.milestones = milestones
                
                # tmp = {scheduler._step_count+n_iter_per_epoch:1 } #* 一个epoch 后decay learning rate 
                # tmp.update({ k:v for  idx, (k,v) in enumerate(scheduler.milestones.items()) if idx != 0})
                # scheduler.milestones = tmp
                
            
            #!=========================================

        # Just eval and end execution
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            return

        # Training loop
        #!===============================
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')
        if osp.exists(save_dir):
            os.remove(save_dir)
        #!===============================
        

        logger.info(scheduler.milestones)
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr']
                )
            )
            # save model
            if epoch % args.val_freq == 0:
                
                # save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Test evaluation.......")
                #!+==========================================
                # self.evaluate_one_epoch(
                #     epoch, test_loader,
                #     model, criterion, set_criterion, args
                # )
                # if dist.get_rank() == 0:
                #     save_checkpoint(args, epoch, model, optimizer, scheduler) 
                performance = self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )

                if dist.get_rank() == 0:
                    if args.upload_wandb:
                        wandb.log(performance)
                        
                    with open(save_dir, 'a+')as f :
                        f.write( f"epoch:{epoch},"+','.join(["%s:%.4f"%(k,v) for k,v in performance.items()])+"\n")
                        
                    acc_key = list(performance.keys())[0]
                    if performance is not None and performance[acc_key] > best_performce:
                        best_performce =  performance[acc_key]
                        save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True)            
                #!+==========================================

        # Training is over, evaluate
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances']
        }

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk
        )
        return loss, end_points


    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict






    def train_one_epoch(self, epoch, train_loader, model,
                        criterion, set_criterion,
                        optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode

        # Loop over batches
        for batch_idx, batch_data in enumerate(train_loader):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            # Forward pass
            end_points = model(inputs)

            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert (key not in end_points)
                end_points[key] = batch_data[key]
            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args
            )


            optimizer.zero_grad()
            loss.backward()
            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                stat_dict['grad_norm'] = grad_total_norm
            optimizer.step()
            scheduler.step()


            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / args.print_freq:.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ]))

                #!==============================================
                # logger.warning(f"epoch : {epoch} ,  lr : {scheduler.get_lr()[0]}")
                if args.upload_wandb and args.local_rank==0:
                    
                    tmp = { f'{key}':stat_dict[key] / args.print_freq  for key in sorted(stat_dict.keys()) if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key }
                    tmp.update({"lr": scheduler.get_last_lr()[0]})
                    

                    wandb.log(tmp)

                #!==============================================

                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0

                
                

            
                

    

    ''' 
    description: 检查输入 
    param {*} self
    param {*} inputs
    return {*}
    '''
    def check_input(self,batch_data):
        #* 132 == MAX_NUM_OBJ ,the max number of object  in that scenes 
        # inputs['point_clouds']#*  [2, 50000, 6]
        # inputs['text'] #* length = 2 
        # inputs['det_boxes']#* [2, 132, 6]
        # inputs['det_bbox_label_mask']#* [2,132]   , 对应是目标还是padding 
        # inputs['det_class_ids']#* [2,132] , 对应类别信息
        inputs = self._get_inputs(batch_data)
        B,OB_NUM=inputs['det_class_ids'].shape
        for i in range(B):
            print(inputs['text'][i])

            # draw_pc_box(
            #             numpy2open3d_colorful(inputs['point_clouds'][i].clone().detach().cpu().numpy()),
            #             inputs['det_boxes'][i][inputs['det_bbox_label_mask'][i]].clone().detach().cpu().numpy(),
            #             save_path=os.path.join(self.vis_save_path, '%s_gt.txt'%(batch_data['scan_ids'][i]))
            #             )  
            
            write_pc_as_ply(
                        inputs['point_clouds'][i].clone().detach().cpu().numpy(),
                        os.path.join(self.vis_save_path, '%s_gt.ply'%(batch_data['scan_ids'][i]))
                    )
                    

            

            #* ply format bounding box 
            # write_oriented_bbox(
            #         inputs['det_boxes'][i][inputs['det_bbox_label_mask'][i]].clone().detach().cpu().numpy(),
            #         os.path.join(self.vis_save_path, '%s_box.ply'%(batch_data['scan_ids'][i])),colors=None
            # )#* colors 可以定义框的颜色


            #* open3d  format bounding box 
            np.savetxt(os.path.join(self.vis_save_path, '%s_box.txt'%(batch_data['scan_ids'][i])),
            inputs['det_boxes'][i][inputs['det_bbox_label_mask'][i]].clone().detach().cpu().numpy(),
            fmt='%s')
            



            #* write utterances
            with open(os.path.join(self.vis_save_path, '%s_utterances.txt'%(batch_data['scan_ids'][i])), 'w') as f :
                f.write(batch_data['utterances'][i])
            


     
    
    '''
    description: 
    return {*}
    '''
    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # Forward pass
        #? what is the output of model 

        #!+============================================
        
        if self.DEBUG:
            self.check_input(batch_data) 
            with open (osp.join(osp.dirname(self.vis_save_path),'current_path.txt'),'w') as f :#* save path
                f.write(self.vis_save_path)
        #!+============================================
        #         
        end_points = model(inputs)#* the length of end_points  == 60, last item ==  last_sem_cls_scores

        # Compute loss
        for key in batch_data: 
            assert (key not in end_points)
            end_points[key] = batch_data[key]#*  the length of end_points == 86, last item ==  target_cid 


        _, end_points = self._compute_loss(#*  the length of end_points == 120
            end_points, criterion, set_criterion, args 
        )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)


        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
            
        return stat_dict, end_points

    
    ''' 
    description: 检查输入 
    param {*} self
    param {*} inputs
    return {*}
    '''
    def check_input(self,inputs,scan_ids,FLAG='student'):
        #* 132 == MAX_NUM_OBJ ,the max number of object  in that scenes 
        # inputs['point_clouds']#*  [2, 50000, 6]
        # inputs['text'] #* length = 2 
        # inputs['det_boxes']#* [2, 132, 6]
        # inputs['det_bbox_label_mask']#* [2,132]   , 对应是目标还是padding 
        # inputs['det_class_ids']#* [2,132] , 对应类别信息

        B,N,_=inputs['point_clouds'].shape
        for i in range(B):

            write_pc_as_ply(
                        inputs['point_clouds'][i].clone().detach().cpu().numpy(),
                        os.path.join(self.vis_save_path,scan_ids[i], '%s_gt_%s.ply'%(scan_ids[i],FLAG))
                    )


            #* open3d  format bounding box 
            np.savetxt(os.path.join(self.vis_save_path, scan_ids[i],'%s_box_%s.txt'%(scan_ids[i],FLAG)),
            inputs['det_boxes'][i][inputs['det_bbox_label_mask'][i]].clone().detach().cpu().numpy(),
            fmt='%s')
            
            #* write utterances
            with open(os.path.join(self.vis_save_path, scan_ids[i],'%s_utterances_%s.txt'%(scan_ids[i],FLAG)), 'w') as f :
                f.write(inputs['text'][i])


     
    '''
    description:  检查target box
    param {*} self
    param {*} batch_data
    param {*} conrresponding_pc
    param {*} scan_ids
    return {*}
    '''
    def check_target(self,batch_data,scan_ids):
        flag = 'target'
        mask = batch_data['box_label_mask']
        gt_box = torch.cat([batch_data['center_label'],batch_data['size_gts']],dim=2)
        for i in range(len(mask)):
            #* open3d  format bounding box 
            np.savetxt(os.path.join(self.vis_save_path, scan_ids[i],'%s_box_%s.txt'%(scan_ids[i],flag)),
                        gt_box[i][mask[i].bool()].clone().detach().cpu().numpy(),fmt='%s')


                
                

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
