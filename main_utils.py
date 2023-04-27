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

# from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from models import HungarianMatcher, SetCriterion, compute_labeled_hungarian_loss
from utils import get_scheduler, setup_logger



from my_utils.vis_utils import *
from my_utils.pc_utils import *

from loguru import logger




import os.path as osp

from IPython import embed
from models.ap_helper import my_parse_predictions
from data.model_util_scannet import ScannetDatasetConfig

#*=====================================
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)
from my_utils.utils import make_dirs, save_res,parse_option,detach_module,load_checkpoint,save_checkpoint

#*=====================================
        
def parse_option():

    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    #* Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    #* Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    #* Loss
    parser.add_argument('--query_points_obj_topk', default=8, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    #* Data
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

    #* Training
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
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-scanrefer', default=False, action='store_true',help=' generate the pred.json for the ')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)
    parser.add_argument('--reduce_lr', action='store_true')

    #* mine args 
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument('--vis-save-path', default=None, type=str)
    parser.add_argument('--wandb',action='store_true', help="upload to wandb or not ?")
    parser.add_argument('--labeled_ratio', default=None, type=float,help=' labeled datasets ratio ')
    parser.add_argument('--use-tkps',action='store_true', help="use-tkps")
    parser.add_argument('--ref_use_obj_mask',action='store_true', help="ref_use_obj_mask")
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



class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):

        self.args = args

        """Initialize."""
        if args.checkpoint_path is not None:
            # name = 'resume'
            name = args.checkpoint_path.split('/')[-4]
            args.log_dir = '/'.join(args.checkpoint_path.split('/')[:-1])
        else:
            name = args.log_dir.split('/')[-1]
            args.log_dir = os.path.join(
                args.log_dir,
                ','.join(args.dataset),
                time.strftime("%Y-%m-%d-%H:%M",time.gmtime(time.time()))
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
            self.log("Full config saved to {}".format(path))
            self.log(str(vars(args)))
            if args.wandb:
                self.init_wandb()
            

    def init_wandb(self):
        if self.args.checkpoint_path is not None:
            #todo resume wandb :
            with open(join(self.args.log_dir,'wandb_resume_info.json'), 'r') as f :
                last_run_info  = json.load(f)
            run  = wandb.init(project='BUTD_DETR',id=last_run_info['id'], resume="must")
            self.log(f"wandb has been resume ")
        else:
            run = wandb.init(project='BUTD_DETR')
            run.name = split(self.args.log_dir)[-1]
            with open(join(self.args.log_dir,'wandb_resume_info.json'),'w') as f :
                json.dump({
                    "id":run.id,
                    "name":run.name,
                },f)
            
            for k, v in self.args.__dict__.items():
                setattr(wandb.config,k,v)
            setattr(wandb.config,"save_root",self.args.log_dir)
            self.log(f"wandb init process has done")
        self.use_wandb = True
    
    def log(self,message):
        if dist.get_rank() == 0 or self.args.local_rank == -1:
            # print((message))
            self.logger.info(message)

        
    def wandb_log(self,message):
        if dist.get_rank() == 0 and hasattr(self,'use_wandb') and self.use_wandb:
            wandb.log(message)


    def get_dataset(self):
        """Initialize datasets."""
        raise NotImplementedError
        # train_dataset = None
        # test_dataset = None
        # return train_dataset, test_dataset


    def seed_worker(self,worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    

    

    '''
    description:  封装一个datasets
    param {*} self
    param {*} datasets
    param {*} shuffle : 如果是test set 就需要等于false
    
    return {*}
    '''
    def get_dataloader(self,datasets,bs,num_works,shuffle=True):
        g = torch.Generator()
        g.manual_seed(0)
        
        data_sampler = DistributedSampler(datasets,shuffle= shuffle)
        dataloader = DataLoader(
            datasets,
            batch_size=bs,
            shuffle=False,
            num_workers=num_works,
            worker_init_fn=self.seed_worker,
            pin_memory=True,
            sampler=data_sampler,
            drop_last=True,
            generator=g
        )
        return dataloader
    

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
        # criterion = compute_hungarian_loss
        criterion= compute_labeled_hungarian_loss

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


    '''
    description: ???
    param {*} self
    param {*} args
    return {*}
    '''
    def evaluation(self,args):
        assert os.path.isfile(args.checkpoint_path)
        if args.checkpoint_path is None :
            logger.error("not checkpoint found ")
            return 



        torch.cuda.set_device(args.local_rank)
        
        """Run main training/testing pipeline."""
        test_dataset = self.get_dataset(args.data_root,{},args.test_dataset,
                    'val' if not args.eval_train else 'train',
                    args.use_color,args.use_height,args.detect_intermediate,
                    args.use_multiview,args.butd,args.butd_gt,
                    args.butd_cls,debug = args.debug,labeled_ratio=args.labeled_ratio)



        test_loader = self.get_dataloader(test_dataset,args.batch_size,args.num_workers,shuffle=False)
        

        # Get model
        model = self.get_model(args)
   

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        
        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda(args.local_rank)

        #* 2.eval and save res to a txt file 
        load_checkpoint(args, model, None, None)



        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )
        
        
        #* file and variable for saving the eval res 
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')
        if osp.exists(save_dir):
            os.remove(save_dir)


        #* eval student model 
        performance = self.evaluate_one_epoch(
            args.start_epoch, test_loader,
            model, criterion, set_criterion, args
        )

        if performance is not None :
            self.log(','.join(['student_%s:%.04f'%(k,round(v,4)) for k,v in performance.items()]))
            is_best,snew_performance = save_res(save_dir,args.start_epoch-1,performance,best_performce)



                

    def main(self, args):

        torch.cuda.set_device(args.local_rank)
        self.log(f"args.local_rank == {args.local_rank}")
        
        
        """Run main training/testing pipeline."""
        # Get loaders

        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1

        if args.joint_det:
            dataset_dict['scannet'] = 10

        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        train_dataset = self.get_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'train' if not args.debug else 'val', 
                        args.use_color,args.use_height,args.detect_intermediate,
                        args.use_multiview,args.butd,args.butd_gt,
                        args.butd_cls,args.augment_det,args.debug,labeled_ratio=args.labeled_ratio)

        train_loader = self.get_dataloader(train_dataset,args.batch_size,args.num_workers,shuffle=True)


        test_dataset = self.get_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'val' if not args.eval_train else 'train',
                         args.use_color,args.use_height,args.detect_intermediate,
                         args.use_multiview,args.butd,args.butd_gt,
                         args.butd_cls,debug = args.debug)


        test_loader = self.get_dataloader(test_dataset,args.batch_size,args.num_workers,shuffle=False)
        

        n_data = len(train_loader.dataset)
        self.log(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.log(f"length of testing dataset: {n_data}")

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

      
        
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cuda(args.local_rank))
        # model = DistributedDataParallel(model,device_ids=[args.local_rank],find_unused_parameters=True,broadcast_buffers = True) 
         
        #* file and variable for saving the eval res 
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')
        if osp.exists(save_dir):
            os.remove(save_dir)


        #* 1.Check for a checkpoint
        #* 2.eval and save res to a txt file 
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)

            #* update lr decay milestones
            #* load model 之后 schedule 也变了 , 变成上次训练的,这次的就不见了, 重新加载
            if args.lr_decay_intermediate:
                
                self.log(f"current step :{scheduler._step_count},last epoch {scheduler.last_epoch} , warm up epoch :{args.warmup_epoch},args.lr_decay_epochs :{args.lr_decay_epochs},len(train_loader):{len(train_loader)}")
                
                
                # tmp = {scheduler._step_count+len(train_loader):1 } #* 一个epoch 后decay learning rate 
                # tmp.update({ k:v for  idx, (k,v) in enumerate(scheduler.milestones.items()) if idx != 0})
                
                scheduler.milestones ={len(train_loader)*( l-args.warmup_epoch - args.start_epoch )+scheduler.last_epoch : 1 for l in args.lr_decay_epochs}
                

            #* eval student model 
            if args.eval:
                performance = self.evaluate_one_epoch(
                    args.start_epoch, test_loader,
                    model, criterion, set_criterion, args
                )
                
                if performance is not None :
                    self.log(','.join(['student_%s:%.04f'%(k,round(v,4)) for k,v in performance.items()]))
                    is_best,snew_performance = save_res(save_dir,args.start_epoch-1,performance,best_performce)

                    if is_best:
                        best_performce = snew_performance
                
                
             
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )

        self.log(scheduler.milestones)
        last_best_epoch_path = None
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )

            self.log(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr']
                )
            )
            # save model
            if epoch % args.val_freq == 0:
            
                print("Test evaluation.......")
                performance = self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )
                
                if dist.get_rank() == 0:
                    if args.upload_wandb:
                        self.wandb_log(performance)
                        
                        
                    with open(save_dir, 'a+')as f :
                        f.write( f"epoch:{epoch},"+','.join(["%s:%.4f"%(k,v) for k,v in performance.items()])+"\n")
                        
                    acc_key = list(performance.keys())[0]
                    if performance is not None and performance[acc_key] > best_performce:
                        best_performce =  performance[acc_key]
                        spath = save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True)
                        self.wandb_log({'Metrics/best_acc':best_performce})
                        

                        if last_best_epoch_path is not None:
                            os.remove(last_best_epoch_path)
                        last_best_epoch_path = spath
    


        # Training is over, evaluate
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)


        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.log("Saved in {}".format(saved_path))
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
            ref_use_obj_mask = args.ref_use_obj_mask,
            query_points_obj_topk=args.query_points_obj_topk,
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
                self.log(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '
                )
                self.log(''.join([
                    f'{key} {stat_dict[key] / args.print_freq:.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ]))

                
                if args.upload_wandb and args.local_rank==0:
                    
                    tmp = { f'Loss/{key}':stat_dict[key] / args.print_freq  for key in sorted(stat_dict.keys()) if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key }
                    tmp.update({"Misc/lr": scheduler.get_last_lr()[0],'Misc/grad_norm':stat_dict['grad_norm'],'epoch':epoch})
                    self.wandb_log(tmp)

                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0
     
    

    
    '''
    description: 
    return {*}
    '''
    def _main_eval_branch(self,batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        #!============================================3. pass the gt information for ploting the qualtative result====================================================
        # assert len(batch_data['scan_ids']) == 1
        # scene_name  = "__".join([batch_data['scan_ids'][0],str(batch_data['target_id'][0].clone().cpu().numpy().tolist()), batch_data['ann_id'][0]])
        # np.savetxt('logs/debug/tmp_name.txt',[scene_name],fmt='%s')
        #!=============================================================================================================================================================

        # Forward pass
        end_points = model(inputs)#* the length of end_points  == 60, last item ==  last_sem_cls_scores

        # Compute loss
        for key in batch_data: 
            assert (key not in end_points)
            end_points[key] = batch_data[key]#*  the length of end_points == 86, last item ==  target_cid 
        #!============================================5. save predict box ===========================================================================================
        # #* according to contrast 
        # CONFIG_DICT = {
        #     'remove_empty_box': False, 'use_3d_nms': True,
        #     'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
        #     'per_class_proposal': True, 'conf_thresh': 0.5,
        #     'dataset_config': ScannetDatasetConfig(18),
        #     'hungarian_loss': True
        # }
        
        # batch_pred_map_cls = my_parse_predictions(end_points,CONFIG_DICT,'last_')
        # #* 已经假设bs == 1 
        # batch_res = batch_pred_map_cls[0]
        # #* 1. 获取target id 
        # #* 2. 根据target id 获取这个target 对应的 score 最大的target  
        # #* 3. 保存对应的 box等信息
        # batch_res=  np.array(batch_res)
        # # target_id = batch_data['target_cid'].cpu().numpy().tolist()[idx]
        # # batch_res = batch_res[batch_res[:,0]==target_id]
        # max_idx = np.argmax(np.array([x[2] for x in batch_res])) #* 只取confidence 最大的, 不管是什么哪个target  , 这个对应的是target id , 也就是第几个目标
        # boxes = np.array([ box.tolist() for box in batch_res[max_idx][1]])[None]
        # write_bbox(boxes,f'logs/debug/scene/{scene_name}/pred_box.ply')
        #!=============================================================================================================================================================

        _, end_points = self._compute_loss(#*  the length of end_points == 120
            end_points, criterion, set_criterion, args 
        )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)


        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.log(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.log(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
            
        return stat_dict, end_points




    
            

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
