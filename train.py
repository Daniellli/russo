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
"""Main script for language modulation."""

import os


import numpy as np
import torch
import torch.distributed as dist
from train_dist_mod import TrainTester
from src.joint_semi_supervise_dataset import JointSemiSupervisetDataset

from src.joint_labeled_dataset import JointLabeledDataset
from src.joint_unlabeled_dataset import JointUnlabeledDataset

import ipdb
st = ipdb.set_trace
import sys 

import wandb
from loguru import logger 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


import os.path as osp
import time
from torch.nn.parallel import DistributedDataParallel
from main_utils import save_checkpoint,load_checkpoint,get_scheduler

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random 
from my_script.consistant_loss import get_consistency_loss
from my_script.utils import parse_semi_supervise_option,save_res,make_dirs,remove_file

from IPython import embed




class SemiSuperviseTrainTester(TrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)
            
            
        

    ''' 
    description:  就是 不是直接用args.consistency_weight  , 而是用这个公式不断靠近这个consistency_weight,E.g.0.1,0.2....10
    param {*} self
    param {*} epoch
    return {*}
    '''
    def get_current_consistency_weight(self,weight,epoch,args):
        
        def sigmoid_rampup(current,args):
            # rampup_length =  args.max_epoch - args.start_epoch +1
            
            rampup_length = args.rampup_length
            if rampup_length == 0:
                return 1
            logger.info(f"rampup_length:{rampup_length}")
            
            current=  current-args.start_epoch
            current = np.clip(current,0,rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))#* initial : 0.007082523

        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return weight * sigmoid_rampup(epoch,args)



            
      
    '''
    description: 获取一个数据集
    data_root: 数据根目录
    train_dataset_dict: 这次用于训练的所有数据集, e.g.: sr3d,nr3d....
    test_datasets: 测试数据集
    split: train or val 
    use_color : 是否使用颜色
    use_height: 是否使用height
    detect_intermediate: 是否对utterance里的名词都作为监督信号,  
    use_multiview : 是否使用多视角数据
    butd:  ??? 好像没用!
    butd_gt :  是否将gt scene box 赋予 detected box 
    butd_cls : 在classification task , 需要将detected box 转成 scene gt box  , 这是这个任务的特点
    augment_det : 是否对detected box 以30% 概率将detected box 替换成 无效的box(随机box)
    debug : 是否overfit 
    return {*}
    '''    
    def get_dataset(self,data_root,train_dataset_dict,test_datasets,split,use_color,use_height,
                    detect_intermediate,use_multiview,butd,butd_gt,butd_cls,
                    augment_det=False,debug=False,labeled_ratio=None):

        return JointLabeledDataset(
            dataset_dict=train_dataset_dict,
            test_dataset=test_datasets,
            split=split,
            use_color=use_color, use_height=use_height,
            overfit=debug,
            data_path=data_root,
            detect_intermediate=detect_intermediate,
            use_multiview=use_multiview,
            butd=butd, 
            butd_gt=butd_gt,
            butd_cls=butd_cls,
            augment_det=augment_det ,
            labeled_ratio=labeled_ratio
        )

    

    '''
    description: 
    unlabel_dataset_root : 是omni supervise 加载 arkitscens 数据集用的
    return {*}
    '''
    def get_unlabeled_dataset(self,data_root,train_dataset_dict,test_datasets,split,use_color,use_height,
                    detect_intermediate,use_multiview,butd,butd_gt,butd_cls,
                    augment_det=False,debug=False,labeled_ratio=None,unlabel_dataset_root=None):

        logger.info(f"unlabeled datasets,ratio {labeled_ratio} , has been loaded ")


        return JointUnlabeledDataset(
            dataset_dict=train_dataset_dict,
            test_dataset=test_datasets,
            split=split,
            use_color=use_color, use_height=use_height,
            overfit=debug,
            data_path=data_root,
            detect_intermediate=detect_intermediate,
            use_multiview=use_multiview,
            butd=butd, 
            butd_gt=butd_gt,
            butd_cls=butd_cls,
            augment_det=augment_det ,
            labeled_ratio=labeled_ratio
        )

    


    @staticmethod
    def _get_teacher_inputs(batch_data):
        return {
            'point_clouds': batch_data['pc_before_aug'].float(),
            'text': batch_data['utterances'],
            "det_boxes": batch_data['teacher_box'],
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],
            "det_class_ids": batch_data['all_detected_class_ids']
        }




            
    '''
    description: transfer the parameter of student model to teacher model 
    param {*} self
    param {*} model: student model 
    param {*} ema_model:teacher model 
    param {*} alpha
    param {*} global_step
    return {*}
    '''
    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        # logger.info(f"alpha:{alpha} ,global_step :{global_step}")
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def train_one_epoch(self, epoch, labeled_loader,unlabeled_loader ,
                        model,ema_model,
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

        total_iteration=len(labeled_loader)
    
        logger.info(f"total_iteration == {total_iteration}")

        center_consistency_weight = self.get_current_consistency_weight(args.center_consistency_weight ,epoch,args)
        size_consistency_weight = self.get_current_consistency_weight(args.size_consistency_weight ,epoch,args)
        token_consistency_weight = self.get_current_consistency_weight(args.token_consistency_weight ,epoch,args)

        query_consistency_weight = self.get_current_consistency_weight(args.query_consistency_weight ,epoch,args)
        text_consistency_weight = self.get_current_consistency_weight(args.text_consistency_weight ,epoch,args)


        logger.info(f"center_consistency_weight  : {center_consistency_weight}")
        logger.info(f"size_consistency_weight  : {size_consistency_weight}")
        logger.info(f"token_consistency_weight  : {token_consistency_weight}")
        logger.info(f"query_consistency_weight  : {query_consistency_weight}")
        logger.info(f"text_consistency_weight  : {text_consistency_weight}")
        unlabeled_loader_iter=iter(unlabeled_loader)

        
        for batch_idx, batch_data in enumerate(labeled_loader):


            try:
                batch_data_unlabeled = next(unlabeled_loader_iter)
            except StopIteration:
                unlabeled_loader_iter = iter(unlabeled_loader)
                batch_data_unlabeled = next(unlabeled_loader_iter)

            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            batch_data_unlabeled = self._to_gpu(batch_data_unlabeled)
            
            for key in batch_data_unlabeled: #* 两个batch 合成一个batch, 
                if  isinstance(batch_data[key],list):
                    batch_data[key] = batch_data[key]+batch_data_unlabeled[key]
                elif  isinstance(batch_data[key],dict):
                    for kkey in batch_data[key]:
                        batch_data[key][kkey] = torch.cat((batch_data[key][kkey], batch_data_unlabeled[key][kkey]), dim=0)
                else:
                    batch_data[key] = torch.cat((batch_data[key], batch_data_unlabeled[key]), dim=0)

            inputs = self._get_inputs(batch_data)
            teacher_input=self._get_teacher_inputs(batch_data)
            
            
            #* check input
            DEBUG=False 
            if  DEBUG and args.local_rank == 0 :
                for scene in batch_data['scan_ids']:
                    make_dirs(osp.join(self.vis_save_path,scene))

                self.check_input(inputs,batch_data['scan_ids'],'student')
                self.check_input(teacher_input,batch_data['scan_ids'],'teacher')
                self.check_target(batch_data,batch_data['scan_ids'])

                if batch_idx == 5:
                    break

            # Forward pass
            end_points = model(inputs)
            with torch.no_grad():
                teacher_end_points = ema_model(teacher_input)   

            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert (key not in end_points)
                end_points[key] = batch_data[key]
                # teacher_end_points[key] = batch_data[key]

            #* add index for knowing  what is labeled which is unlabeled 
            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args
            )

            end_points = get_consistency_loss(end_points, teacher_end_points,batch_data['augmentations'])
            consistent_loss =center_consistency_loss=soft_token_consistency_loss=size_consistency_loss=query_consistency_loss=text_consistency_loss=None

            center_consistency_loss = end_points['center_consistency_loss'] * center_consistency_weight
            soft_token_consistency_loss = end_points['soft_token_consistency_loss']* token_consistency_weight
            size_consistency_loss = end_points['size_consistency_loss'] * size_consistency_weight
            query_consistency_loss = end_points['query_consistency_loss'] * query_consistency_weight
            text_consistency_loss = end_points['text_consistency_loss'] * text_consistency_weight

            consistent_loss = soft_token_consistency_loss +center_consistency_loss+size_consistency_loss+query_consistency_loss+text_consistency_loss


            #* total loss
            if consistent_loss is not None:
                total_loss = loss+consistent_loss
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            
            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                stat_dict['grad_norm'] = grad_total_norm

            optimizer.step()
            scheduler.step()

            #*===================================================
            #* update  teacher model 
            #* epoch start from 1 by default , so have to minus one 
            global_step = (batch_idx+1) + (epoch -args.start_epoch) *total_iteration
            alpha = args.ema_decay

        

            ran_epoch =  epoch -args.start_epoch
            if ran_epoch>args.rampup_length:
                alpha=args.ema_decay_after_rampup
                


            self.update_ema_variables(model,ema_model,alpha,global_step)
            #*===================================================

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:

                logger.info(f"ran_epoch:{ran_epoch},alpha:{alpha}")
                
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{total_iteration}]  '
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / args.print_freq:.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ]))


                if args.upload_wandb and args.local_rank==0:
                    tmp = { f'{key}':stat_dict[key] / args.print_freq  for key in sorted(stat_dict.keys()) if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key }
                    tmp.update({"student_supervised_loss":loss.clone().detach().item(),
                                "center_consistency_loss":center_consistency_loss.clone().detach().item() if center_consistency_loss is not None else None,
                                "soft_token_consistency_loss":soft_token_consistency_loss.clone().detach().item() if soft_token_consistency_loss is not None else None,
                                "size_consistency_loss":size_consistency_loss.clone().detach().item() if size_consistency_loss is not None else None,
                                "query_consistency_loss":query_consistency_loss.clone().detach().item() if query_consistency_loss is not None else None,
                                "text_consistency_loss":text_consistency_loss.clone().detach().item() if text_consistency_loss is not None else None,
                                "consistent_loss":consistent_loss.clone().detach().item() if consistent_loss is not None else None ,
                                "total_loss":total_loss.clone().detach().item(),
                                "lr": scheduler.get_last_lr()[0]
                            })
                    wandb.log(tmp)

                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0



    def full_supervise_train_one_epoch(self, epoch, labeled_loader,
                        model,ema_model,
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

        total_iteration=len(labeled_loader)
    
        logger.info(f"total_iteration == {total_iteration}")

        center_consistency_weight = self.get_current_consistency_weight(args.center_consistency_weight ,epoch,args)
        size_consistency_weight = self.get_current_consistency_weight(args.size_consistency_weight ,epoch,args)
        token_consistency_weight = self.get_current_consistency_weight(args.token_consistency_weight ,epoch,args)

        query_consistency_weight = self.get_current_consistency_weight(args.query_consistency_weight ,epoch,args)
        text_consistency_weight = self.get_current_consistency_weight(args.text_consistency_weight ,epoch,args)


        logger.info(f"center_consistency_weight  : {center_consistency_weight}")
        logger.info(f"size_consistency_weight  : {size_consistency_weight}")
        logger.info(f"token_consistency_weight  : {token_consistency_weight}")
        logger.info(f"query_consistency_weight  : {query_consistency_weight}")
        logger.info(f"text_consistency_weight  : {text_consistency_weight}")



        
        for batch_idx, batch_data in enumerate(labeled_loader):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)

            inputs = self._get_inputs(batch_data)
            teacher_input=self._get_teacher_inputs(batch_data)
            
            
            # Forward pass
            end_points = model(inputs)
            with torch.no_grad():
                teacher_end_points = ema_model(teacher_input)   

            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert (key not in end_points)
                end_points[key] = batch_data[key]
                # teacher_end_points[key] = batch_data[key]

            #* add index for knowing  what is labeled which is unlabeled 
            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args
            )

            end_points = get_consistency_loss(end_points, teacher_end_points,batch_data['augmentations'])
            consistent_loss =center_consistency_loss=soft_token_consistency_loss=size_consistency_loss=query_consistency_loss=text_consistency_loss=None

            center_consistency_loss = end_points['center_consistency_loss'] * center_consistency_weight
            soft_token_consistency_loss = end_points['soft_token_consistency_loss']* token_consistency_weight
            size_consistency_loss = end_points['size_consistency_loss'] * size_consistency_weight
            query_consistency_loss = end_points['query_consistency_loss'] * query_consistency_weight
            text_consistency_loss = end_points['text_consistency_loss'] * text_consistency_weight

            consistent_loss = soft_token_consistency_loss +center_consistency_loss+size_consistency_loss+query_consistency_loss+text_consistency_loss


            #* total loss
            if consistent_loss is not None:
                total_loss = loss+consistent_loss
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            
            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                stat_dict['grad_norm'] = grad_total_norm

            optimizer.step()
            scheduler.step()

            #*===================================================
            #* update  teacher model 
            #* epoch start from 1 by default , so have to minus one 
            global_step = (batch_idx+1) + (epoch -args.start_epoch) *total_iteration
            alpha = args.ema_decay
            self.update_ema_variables(model,ema_model,alpha,global_step)
            #*===================================================

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{total_iteration}]  '
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / args.print_freq:.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ]))


                if args.upload_wandb and args.local_rank==0:
                    tmp = { f'{key}':stat_dict[key] / args.print_freq  for key in sorted(stat_dict.keys()) if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key }
                    tmp.update({"student_supervised_loss":loss.clone().detach().item(),
                                "center_consistency_loss":center_consistency_loss.clone().detach().item() if center_consistency_loss is not None else None,
                                "soft_token_consistency_loss":soft_token_consistency_loss.clone().detach().item() if soft_token_consistency_loss is not None else None,
                                "size_consistency_loss":size_consistency_loss.clone().detach().item() if size_consistency_loss is not None else None,
                                "query_consistency_loss":query_consistency_loss.clone().detach().item() if query_consistency_loss is not None else None,
                                "text_consistency_loss":text_consistency_loss.clone().detach().item() if text_consistency_loss is not None else None,
                                "consistent_loss":consistent_loss.clone().detach().item() if consistent_loss is not None else None ,
                                "total_loss":total_loss.clone().detach().item(),
                                "lr": scheduler.get_last_lr()[0]
                            })
                    wandb.log(tmp)

                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0


    def main(self, args):
        #!======================= 避免数据跑到其他卡上
        torch.cuda.set_device(args.local_rank)
        logger.info(f"args.local_rank == {args.local_rank}")
        #!=======================

        """Run main training/testing pipeline."""
        # Get loaders
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1

        if args.joint_det:
            dataset_dict['scannet'] = 10



        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        labeled_dataset = self.get_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'train' if not args.debug else 'val', 
                        args.use_color,args.use_height,args.detect_intermediate,
                        args.use_multiview,args.butd,args.butd_gt,
                        args.butd_cls,args.augment_det,args.debug,
                        labeled_ratio=args.labeled_ratio)


        #* 可能根据labeled_ratio 也可能根据unlabel_dataset_root 加载arkitscenes 
        #* 取决于 是被train.py 调用还是  omni_supervise_train.py 调用
        unlabeled_datasets  = self.get_unlabeled_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'train' if not args.debug else 'val', 
                        args.use_color,args.use_height,args.detect_intermediate,
                        args.use_multiview,args.butd,args.butd_gt,
                        args.butd_cls,args.augment_det,args.debug,
                        labeled_ratio=args.labeled_ratio,unlabel_dataset_root=args.unlabel_dataset_root)
                
        
        test_dataset = self.get_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'val' if not args.eval_train else 'train',
                         args.use_color,args.use_height,args.detect_intermediate,
                         args.use_multiview,args.butd,args.butd_gt,
                         args.butd_cls,debug = args.debug)




        batch_size_list = np.array(args.batch_size.split(',')).astype(np.int64)
        labeled_loader = self.get_dataloader(labeled_dataset,int(batch_size_list[0]),args.num_workers,shuffle = True)
        unlabeled_loader = self.get_dataloader(unlabeled_datasets,int(batch_size_list[1]),args.num_workers,shuffle = True)
        test_loader = self.get_dataloader(test_dataset,int(batch_size_list.sum().astype(np.int64)),args.num_workers,shuffle = False)
        logger.info(f"un supervised mask :{unlabeled_loader.dataset.__getitem__(0)['supervised_mask']}")
        


        logger.info(f"length of  labeled dataset: {len(labeled_loader.dataset)} \t  length of  unlabeled dataset: {len(unlabeled_loader.dataset)} \t length of testing dataset: {len(test_loader.dataset)}")
        
        # Get model
        model = self.get_model(args)
        ema_model = self.get_model(args)
        
        for param in ema_model.parameters():
            param.detach_()

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)


        # Get scheduler
        scheduler = get_scheduler(optimizer, len(labeled_loader), args)#* 第二个参数是一个epoch需要iteration 多少次 

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda(args.local_rank)
            ema_model = ema_model.cuda(args.local_rank)

        
        #* file and variable for saving the eval res 
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')

        ema_best_performce = 0
        ema_save_dir = osp.join(args.log_dir,'ema_performance.txt')

        if osp.exists(save_dir):
            os.remove(save_dir)

        #* 1.Check for a checkpoint
        #* 2.eval and save res to a txt file 
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            load_checkpoint(args, ema_model, optimizer, scheduler)

            #* update lr decay milestones
            if args.lr_decay_intermediate:    
                # tmp = {scheduler._step_count+len(labeled_loader):1 } #* 一个epoch 后decay learning rate 
                # tmp.update({ k:v for  idx, (k,v) in enumerate(scheduler.milestones.items()) if idx != 0})
                # scheduler.milestones = tmp
                logger.info(f"scheduler._step_count :{scheduler._step_count},args.start_epoch:{args.start_epoch},args.warmup_epoch:{args.warmup_epoch}")
                decay_epoch = [( l-args.warmup_epoch - args.start_epoch ) for l in args.lr_decay_epochs]

                scheduler.milestones ={len(labeled_loader)*( l-args.warmup_epoch - args.start_epoch )+scheduler.last_epoch : 1 for l in args.lr_decay_epochs}
                logger.info(scheduler.milestones )

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

                #* eval teacher model 
                ema_performance = self.evaluate_one_epoch(
                    args.start_epoch, test_loader,
                    ema_model, criterion, set_criterion, args
                )

                if ema_performance is not None :
                    logger.info(','.join(['teacher_%s:%.04f'%(k,round(v,4)) for k,v in ema_performance.items()]))
                    is_best,tnew_performance = save_res(ema_save_dir,args.start_epoch-1,ema_performance,ema_best_performce)
                    if is_best:
                        ema_best_performce= tnew_performance
                exit(0)


        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )
                    
        #* Training loop
        last_student_best_path = last_teacher_best_path = None
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            
            labeled_loader.sampler.set_epoch(epoch)
            unlabeled_loader.sampler.set_epoch(epoch)


            tic = time.time()
            

            self.train_one_epoch(
                epoch, labeled_loader, unlabeled_loader,model,ema_model,
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
                
                if performance is not None :
                    logger.info(','.join(['student_%s:%.04f'%(k,round(v,4)) for k,v in performance.items()]))
                    is_best,snew_performance = save_res(save_dir,epoch,performance,best_performce)
                    if is_best:
                        best_performce =  snew_performance
                        spath = save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True,prefix='student_')

                        remove_file(last_student_best_path)
                        last_student_best_path = spath

                
                ema_performance = self.evaluate_one_epoch(
                    epoch, test_loader,
                    ema_model, criterion, set_criterion, args
                )

                if ema_performance is not None :
                    logger.info(','.join(['teacher_%s:%.04f'%(k,round(v,4)) for k,v in ema_performance.items()]))
                    ema_is_best,tnew_performance = save_res(ema_save_dir,epoch,ema_performance,ema_best_performce)
                    if ema_is_best:
                        ema_best_performce =  tnew_performance
                        spath= save_checkpoint(args, epoch, ema_model, optimizer, scheduler ,is_best=True,prefix='teacher_')
                
                        remove_file(last_teacher_best_path)
                        last_teacher_best_path = spath
                        
                
                
                #todo 把save as txt 分离出来? 
                if dist.get_rank() == 0 and args.upload_wandb:
                    #* model (student model )
                    if performance is not None :
                        wandb.log({'student_%s'%(k):round(v,4) for k,v in performance.items()})
                        if is_best:
                            wandb.log({'%s'%('student_best_'+k):round(v,4) for k,v in performance.items()})

                    if ema_performance is not None :
                        wandb.log({'teacher_%s'%(k):round(v,4) for k,v in ema_performance.items()})
                        if ema_is_best:
                            wandb.log({'%s'%('teacher_best_'+k):round(v,4) for k,v in ema_performance.items()})


        # Training is over, evaluate
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path

    
    '''
    description:  全部使用有标签数据过EMA architecture
    param {*} self
    param {*} args
    return {*}
    '''
    def full_supervise_main(self, args):
        #!======================= 避免数据跑到其他卡上
        torch.cuda.set_device(args.local_rank)
        logger.info(f"args.local_rank == {args.local_rank}")
        #!=======================

        """Run main training/testing pipeline."""
        # Get loaders
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1

        if args.joint_det:
            dataset_dict['scannet'] = 10



        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        labeled_dataset = self.get_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'train' if not args.debug else 'val', 
                        args.use_color,args.use_height,args.detect_intermediate,
                        args.use_multiview,args.butd,args.butd_gt,
                        args.butd_cls,args.augment_det,args.debug,
                        labeled_ratio=None)#* using all  training datasets  as labeled datasets 


        
        test_dataset = self.get_dataset(args.data_root,dataset_dict,args.test_dataset,
                        'val' if not args.eval_train else 'train',
                         args.use_color,args.use_height,args.detect_intermediate,
                         args.use_multiview,args.butd,args.butd_gt,
                         args.butd_cls,debug = args.debug)




        batch_size_list = np.array(args.batch_size.split(',')).astype(np.int64)


        labeled_loader = self.get_dataloader(labeled_dataset,int(batch_size_list.sum()),args.num_workers,shuffle = True)
        test_loader = self.get_dataloader(test_dataset,int(batch_size_list.sum().astype(np.int64)),args.num_workers,shuffle = False)


        logger.info(f"length of  labeled dataset: {len(labeled_loader.dataset)} \t length of testing dataset: {len(test_loader.dataset)}")

        
        # Get model
        model = self.get_model(args)
        ema_model = self.get_model(args)
        
        for param in ema_model.parameters():
            param.detach_()

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)


        # Get scheduler
        scheduler = get_scheduler(optimizer, len(labeled_loader), args)#* 第二个参数是一个epoch需要iteration 多少次 

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda(args.local_rank)
            ema_model = ema_model.cuda(args.local_rank)

       

        
        #* file and variable for saving the eval res 
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')

        ema_best_performce = 0
        ema_save_dir = osp.join(args.log_dir,'ema_performance.txt')

        if osp.exists(save_dir):
            os.remove(save_dir)

        #* 1.Check for a checkpoint
        #* 2.eval and save res to a txt file 
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            load_checkpoint(args, ema_model, optimizer, scheduler)

            #* update lr decay milestones
            if args.lr_decay_intermediate:    
                logger.info(f"scheduler._step_count :{scheduler._step_count},args.start_epoch:{args.start_epoch},args.warmup_epoch:{args.warmup_epoch}")
                decay_epoch = [( l-args.warmup_epoch - args.start_epoch ) for l in args.lr_decay_epochs]

                scheduler.milestones ={len(labeled_loader)*( l-args.warmup_epoch - args.start_epoch )+scheduler.last_epoch : 1 for l in args.lr_decay_epochs}
                logger.info(scheduler.milestones )

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

                #* eval teacher model 
                ema_performance = self.evaluate_one_epoch(
                    args.start_epoch, test_loader,
                    ema_model, criterion, set_criterion, args
                )

                if ema_performance is not None :
                    logger.info(','.join(['teacher_%s:%.04f'%(k,round(v,4)) for k,v in ema_performance.items()]))
                    is_best,tnew_performance = save_res(ema_save_dir,args.start_epoch-1,ema_performance,ema_best_performce)
                    if is_best:
                        ema_best_performce= tnew_performance
                exit(0)


        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )

                    
        #* Training loop
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            
            labeled_loader.sampler.set_epoch(epoch)


            tic = time.time()
            

            self.full_supervise_train_one_epoch(
                epoch, labeled_loader,model,ema_model,
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
                
                if performance is not None :
                    logger.info(','.join(['student_%s:%.04f'%(k,round(v,4)) for k,v in performance.items()]))
                    is_best,snew_performance = save_res(save_dir,epoch,performance,best_performce)
                    if is_best:
                        best_performce =  snew_performance
                        save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True,prefix='student_')

                
                ema_performance = self.evaluate_one_epoch(
                    epoch, test_loader,
                    ema_model, criterion, set_criterion, args
                )

                if ema_performance is not None :
                    logger.info(','.join(['teacher_%s:%.04f'%(k,round(v,4)) for k,v in ema_performance.items()]))
                    ema_is_best,tnew_performance = save_res(ema_save_dir,epoch,ema_performance,ema_best_performce)
                    if ema_is_best:
                        ema_best_performce =  tnew_performance
                        save_checkpoint(args, epoch, ema_model, optimizer, scheduler ,is_best=True,prefix='teacher_')     
                
                
                #todo 把save as txt 分离出来? 
                if dist.get_rank() == 0 and args.upload_wandb:
                    #* model (student model )
                    if performance is not None :
                        wandb.log({'student_%s'%(k):round(v,4) for k,v in performance.items()})
                        if is_best:
                            wandb.log({'%s'%('student_best_'+k):round(v,4) for k,v in performance.items()})

                    if ema_performance is not None :
                        wandb.log({'teacher_%s'%(k):round(v,4) for k,v in ema_performance.items()})
                        if ema_is_best:
                            wandb.log({'%s'%('teacher_best_'+k):round(v,4) for k,v in ema_performance.items()})


        # Training is over, evaluate
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path




    
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    opt = parse_semi_supervise_option()
    
    # logger.info(f"gpu ids == {opt.gpu_ids}")
    # logger.info(os.environ["CUDA_VISIBLE_DEVICES"] )
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.distributed.init_process_group(backend='nccl')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.cuda.set_device(opt.local_rank)
    train_tester = SemiSuperviseTrainTester(opt)
    if opt.upload_wandb and opt.local_rank==0:
        run=wandb.init(project="BUTD_DETR")
        run.name = "test_"+run.name
        for k, v in opt.__dict__.items():
            setattr(wandb.config,k,v)



    if opt.ema_full_supervise:
        ckpt_path = train_tester.full_supervise_main(opt)
    else:
        ckpt_path = train_tester.main(opt)
    
    