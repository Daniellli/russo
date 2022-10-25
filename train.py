'''
Author: xushaocong
Date: 2022-10-03 22:00:15
LastEditTime: 2022-10-25 20:25:32
LastEditors: xushaocong
Description:  修改get_datasets , 换成可以添加使用数据集比例的dataloader
FilePath: /butd_detr/train.py
email: xushaocong@stu.xmu.edu.cn
'''
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
from my_script.utils import parse_semi_supervise_option,save_res,make_dirs


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
            rampup_length = 30
            current=  current-args.start_epoch
            current = np.clip(current,0,rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))#* initial : 0.007082523

        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return weight * sigmoid_rampup(epoch,args)


    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1

        if args.joint_det:
            dataset_dict['scannet'] = 10

        # labeled_ratio = 0.2
        # logger.info(f"labeled_ratio:{labeled_ratio}")
        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        
        labeled_dataset = JointLabeledDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset, #? only test set need ? 
            split='train' ,
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,#? 
            use_multiview=args.use_multiview, #? 
            butd=args.butd, #? 
            butd_gt=args.butd_gt,#? 
            butd_cls=args.butd_cls,#? 
            augment_det=args.augment_det,#? 
            labeled_ratio=args.labeled_ratio
        )
        
        
        unlabeled_dataset = JointUnlabeledDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset, #? only test set need ? 
            split='train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,#? 
            use_multiview=args.use_multiview, #? 
            butd=args.butd, #? 
            butd_gt=args.butd_gt,#? 
            butd_cls=args.butd_cls,#? 
            augment_det=args.augment_det,#? 
            labeled_ratio=args.labeled_ratio
        )
        
        test_dataset = JointSemiSupervisetDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls
        )

        
        return labeled_dataset,unlabeled_dataset, test_dataset
        

    @staticmethod
    def _get_teacher_inputs(batch_data):
        return {
            'point_clouds': batch_data['pc_before_aug'].float(),
            'text': batch_data['utterances'],
            "det_boxes": batch_data['teacher_box'],
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],
            "det_class_ids": batch_data['all_detected_class_ids']
        }



    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        #* do not need to load train set when args.evals == True 
        # Datasets
        labeled_dataset,unlabeled_dataset, test_dataset = self.get_datasets(args)
        
        #* 存在一个问题就是val 的数据抽取的不合法,在group_free_pred_bboxes_val 找不到对应的数据
        
        g = torch.Generator()
        g.manual_seed(0)

        batch_size_list = np.array(args.batch_size.split(',')).astype(np.int64)

        labeled_sampler = DistributedSampler(labeled_dataset)
        labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=int(batch_size_list[0]),
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=labeled_sampler,
            drop_last=True,
            generator=g
        )

        unlabeled_sampler = DistributedSampler(unlabeled_dataset)
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=int(batch_size_list[1]),
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=unlabeled_sampler,
            drop_last=True,
            generator=g
        )
        

        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(batch_size_list.sum()),
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        logger.info(f"the iter num  labeled_loader needs :{len(labeled_loader)}, the iter num  unlabeled_loader needs :{len(unlabeled_loader)}, the iter num  test_loader needs :{len(test_loader)},  ")
        return labeled_loader,unlabeled_loader, test_loader


            
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
        labeled_loader,unlabeled_loader, test_loader = self.get_loaders(args)
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

        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )

        
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
            load_checkpoint(args, ema_model, optimizer, scheduler,distributed2common=True)

            #* update lr decay milestones
            if args.lr_decay_intermediate:    
                tmp = {scheduler._step_count+len(labeled_loader):1 } #* 一个epoch 后decay learning rate 
                tmp.update({ k:v for  idx, (k,v) in enumerate(scheduler.milestones.items()) if idx != 0})
                scheduler.milestones = tmp

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



                    
        #* Training loop
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
                        save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True,prefix='student_')

                
                ema_performance = self.evaluate_one_epoch(
                    epoch, test_loader,
                    ema_model, criterion, set_criterion, args
                )

                if ema_performance is not None :
                    logger.info(','.join(['teacher_%s:%.04f'%(k,round(v,4)) for k,v in ema_performance.items()]))
                    is_best,tnew_performance = save_res(ema_save_dir,epoch,ema_performance,ema_best_performce)
                    if is_best:
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
                        if is_best:
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

    ckpt_path = train_tester.main(opt)
    
    