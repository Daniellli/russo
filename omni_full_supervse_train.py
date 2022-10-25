'''
Author: xushaocong
Date: 2022-10-03 22:00:15
LastEditTime: 2022-10-25 18:27:51
LastEditors: xushaocong
Description:  将 ARKitScenes 也作为有标签数据进行监督 , 也就是用mean teacher 

1.和其他main function 不同的是需要 加载labeledARKitScenes ,
2.  然后loss 计算需要考虑 query loss  ,   
3. batch size 需要考虑 所有的labeled datsets 和ARKitScenes datasets , 和这些一样
- train.py 
- omni_supervise_train.py

FilePath: /butd_detr/omni_full_supervse_train.py
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
from src.labeled_arkitscenes_dataset import LabeledARKitSceneDataset


import ipdb
st = ipdb.set_trace
import sys 

import wandb
from loguru import logger 
from my_script.utils import * 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import os.path as osp
import time


from my_script.utils import save_res
from torch.nn.parallel import DistributedDataParallel
from main_utils import save_checkpoint,load_checkpoint,get_scheduler

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random 



class OmniFullSuperviseTrainTester(TrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

  
            
      

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
        
        labeled_dataset = JointSemiSupervisetDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls
        )

        arkitscene_datasets = LabeledARKitSceneDataset(augment=True,data_root='datasets/arkitscenes', butd_cls=args.butd_cls)
        
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

        
        return labeled_dataset,arkitscene_datasets, test_dataset


    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        #* do not need to load train set when args.evals == True 
        # Datasets
        labeled_dataset,arkitscenes_dataset, test_dataset = self.get_datasets(args)
        
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

        arkitscenes_sampler = DistributedSampler(arkitscenes_dataset)
        arkitscenes_loader = DataLoader(
            arkitscenes_dataset,
            batch_size=int(batch_size_list[1]),
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=arkitscenes_sampler,
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
        logger.info(f"the iter num  labeled_loader needs :{len(labeled_loader)}, the iter num  arkitscenes_loader needs :{len(arkitscenes_loader)}, the iter num  test_loader needs :{len(test_loader)},  ")
        return labeled_loader,arkitscenes_loader, test_loader


    


    def train_one_epoch(self, epoch, labeled_loader,arkitscene_loader ,
                        model,criterion, set_criterion,
                        optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode
        total_iteration = len(labeled_loader)
        arkitscenes_loader_iter=iter(arkitscene_loader)
        
        for batch_idx, batch_data in enumerate(labeled_loader):
            try:
                batch_data_arkitscene = next(arkitscenes_loader_iter)
            except StopIteration:
                arkitscenes_loader_iter=iter(arkitscene_loader)
                batch_data_arkitscene = next(arkitscenes_loader_iter)

            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            batch_data_arkitscene = self._to_gpu(batch_data_arkitscene)
            
            for key in batch_data_arkitscene: #* 两个batch 合成一个batch, 
                if  isinstance(batch_data[key],list):
                    batch_data[key] = batch_data[key]+batch_data_arkitscene[key]
                elif  isinstance(batch_data[key],dict):
                    for kkey in batch_data[key]:
                        batch_data[key][kkey] = torch.cat((batch_data[key][kkey], batch_data_arkitscene[key][kkey]), dim=0)
                else:
                    batch_data[key] = torch.cat((batch_data[key], batch_data_arkitscene[key]), dim=0)

            inputs = self._get_inputs(batch_data)

            # Forward pass
            end_points = model(inputs)
            
            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert (key not in end_points)
                end_points[key] = batch_data[key]
            
            #* add index for knowing  what is labeled which is unlabeled 
            loss, end_points = self._compute_loss(
                end_points, criterion, set_criterion, args
            )

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
                                "total_loss":total_loss.clone().detach().item(),
                                "lr": scheduler.get_last_lr()[0]
                            })
                    wandb.log(tmp)
                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0



    '''
    description: 需要多加载一个dataloader
    param {*} self
    param {*} args
    return {*}
    '''
    def main(self, args):

        torch.cuda.set_device(args.local_rank)
        """Run main training/testing pipeline."""
        # Get loaders
        labeled_loader,arkitscene_loader, test_loader = self.get_loaders(args)
        logger.info(f"length of  labeled dataset: {len(labeled_loader.dataset)} \n  length of  unlabeled dataset: {len(arkitscene_loader.dataset)} \n length of testing dataset: {len(test_loader.dataset)}")
        # Get model
        model = self.get_model(args)
        
        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)


        # Get scheduler
        scheduler = get_scheduler(optimizer, len(labeled_loader), args)#* 第二个参数是一个epoch需要iteration 多少次 

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

        #* 1.Check for a checkpoint
        #* 2.eval and save res to a txt file 
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)

            #* update lr decay milestones
            #* load model 之后 schedule 也变了 , 变成上次训练的,这次的就不见了, 重新加载
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
             
      
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            labeled_loader.sampler.set_epoch(epoch)
            arkitscene_loader.sampler.set_epoch(epoch)
            tic = time.time()

            self.train_one_epoch(
                epoch, labeled_loader, arkitscene_loader,model,
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
      
                #todo 把save as txt 分离出来? 
                if dist.get_rank() == 0 and args.upload_wandb:
                    #* model (student model )
                    
                    if performance is not None :
                        wandb.log({'student_%s'%(k):round(v,4) for k,v in performance.items()})
                        is_best,new_performance = save_res(save_dir,epoch,performance,best_performce)
                        if is_best:
                            best_performce =  new_performance
                            save_checkpoint(args, epoch, model, optimizer, scheduler ,is_best=True,prefix='student_')
                            wandb.log({'%s'%('student_best_'+k):round(v,4) for k,v in performance.items()})


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
    
    train_tester = OmniFullSuperviseTrainTester(opt)

    if opt.upload_wandb and opt.local_rank==0:
        run=wandb.init(project="BUTD_DETR")
        run.name = "test_"+run.name
        for k, v in opt.__dict__.items():
            setattr(wandb.config,k,v)

    ckpt_path = train_tester.main(opt)
    
    