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



# from my_script.vis_utils import *
from my_script.pc_utils import *

from loguru import logger




import os.path as osp

from IPython import embed


#*=====================================
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)
from my_script.utils import make_dirs, save_res,parse_option,detach_module,load_checkpoint,save_checkpoint

#*=====================================

        


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


    def evaluation(self,args):

        assert os.path.isfile(args.checkpoint_path)
        if args.checkpoint_path is None :
            logger.error("not checkpoint found ")
            return 



        torch.cuda.set_device(args.local_rank)
        logger.info(f"args.local_rank == {args.local_rank}")


        
        """Run main training/testing pipeline."""
        if args.eval_scanrefer:
            test_dataset = self.get_scanrefer_dataset(args.data_root,{},args.test_dataset,
                            'test',
                            args.use_color,args.use_height,args.detect_intermediate,
                            args.use_multiview,args.butd,args.butd_gt,
                            args.butd_cls,debug = args.debug,labeled_ratio=args.labeled_ratio)
        else :
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
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )
        
        
        #* file and variable for saving the eval res 
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')
        if osp.exists(save_dir):
            os.remove(save_dir)

        #* 2.eval and save res to a txt file 
        load_checkpoint(args, model, None, None)


        #* eval student model 
        #!==========

        DEBUG = False
        if DEBUG:
            performance = self.inference_for_scanrefer_benchmark(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args,for_vis=False,debug=DEBUG
            )
        else:
            performance = self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )

        if performance is not None :
            logger.info(','.join(['student_%s:%.04f'%(k,round(v,4)) for k,v in performance.items()]))
            is_best,snew_performance = save_res(save_dir,args.start_epoch-1,performance,best_performce)



                

    def main(self, args):

        torch.cuda.set_device(args.local_rank)
        logger.info(f"args.local_rank == {args.local_rank}")
        
        
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
                
                logger.info(f"current step :{scheduler._step_count},last epoch {scheduler.last_epoch} , warm up epoch :{args.warmup_epoch},args.lr_decay_epochs :{args.lr_decay_epochs},len(train_loader):{len(train_loader)}")
                
                
                
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
                    logger.info(','.join(['student_%s:%.04f'%(k,round(v,4)) for k,v in performance.items()]))
                    is_best,snew_performance = save_res(save_dir,args.start_epoch-1,performance,best_performce)

                    if is_best:
                        best_performce = snew_performance
                
                
             
        

        logger.info(scheduler.milestones)
        last_best_epoch_path = None
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
            
                print("Test evaluation.......")
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
                        spath = save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True)            

                        if last_best_epoch_path is not None:
                            os.remove(last_best_epoch_path)
                        last_best_epoch_path = spath
    


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

                
                if args.upload_wandb and args.local_rank==0:
                    
                    tmp = { f'{key}':stat_dict[key] / args.print_freq  for key in sorted(stat_dict.keys()) if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key }
                    tmp.update({"lr": scheduler.get_last_lr()[0]})
                    wandb.log(tmp)

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

        # Forward pass
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
    description:  with debug 
    return {*}
    '''
    def _main_eval_branch_debug(self, batch_idx, batch_data, test_loader, model,
                        stat_dict,criterion, set_criterion, args,debug):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # Forward pass
        #todo 如何把debug 信息传给 model 里面的 DKS? 
        end_points = model(inputs)#* the length of end_points  == 60, last item ==  last_sem_cls_scores

        # Compute loss
        for key in batch_data: 
            assert (key not in end_points)
            end_points[key] = batch_data[key]#*  the length of end_points == 86, last item ==  target_cid 


        #!==================================================      
        #* 1. rename file 
        if  debug:
            # self.check_input(inputs,end_points['scan_ids'])
            prefixes = ['object','text']
            debug_path = "logs/debug"
            save_format='%s_tmp_%d.ply'
            new_save_format='%s_%s_%d_%d.ply'

            for prefix in prefixes:
                print(prefix)
                for idx, scan_name in enumerate(end_points['scan_ids']):
                    target_save_path = osp.join(debug_path,scan_name+"_%d_%d"%(idx,batch_idx))
                    make_dirs(target_save_path)

                    new_name = osp.join(target_save_path, new_save_format%(prefix,scan_name,idx,batch_idx))
                    old_name = osp.join(debug_path, save_format%(prefix,idx))
                    os.rename(old_name,new_name)
        #!==================================================

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
    description:  with debug 
    return {*}
    '''
    def _inference_only(self, batch_idx, batch_data, test_loader, model,
                        stat_dict,criterion, set_criterion, args,debug):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # Forward pass
        #todo 如何把debug 信息传给 model 里面的 DKS? 
        end_points = model(inputs)#* the length of end_points  == 60, last item ==  last_sem_cls_scores

        # Compute loss
        for key in batch_data: 
            assert (key not in end_points)
            end_points[key] = batch_data[key]#*  the length of end_points == 86, last item ==  target_cid 

            
        return end_points

    
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
