'''
Author: xushaocong
Date: 2022-10-03 22:00:15
LastEditTime: 2022-10-25 18:26:12
LastEditors: xushaocong
Description:  修改get_datasets , 换成可以添加使用数据集比例的dataloader
FilePath: /butd_detr/pretrain.py
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

from main_utils import parse_option, BaseTrainTester
from train_dist_mod import TrainTester
from src.joint_labeled_dataset import JointLabeledDataset
from src.joint_semi_supervise_dataset import JointSemiSupervisetDataset





from IPython import embed
import ipdb
st = ipdb.set_trace
import scipy.io as scio
import sys 

from utils.box_util import get_3d_box
import json
import wandb
from loguru import logger 
from my_script.utils import parse_option


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



import os.path as osp
import time


class PretrainTrainTester(TrainTester):
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

        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        train_dataset = JointLabeledDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset, #? only test set need ? 
            split='train' if not args.debug else 'val',
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
        
        return train_dataset, test_dataset



if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    opt = parse_option()
    
    # logger.info(f"gpu ids == {opt.gpu_ids}")
    # logger.info(os.environ["CUDA_VISIBLE_DEVICES"] )
     
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.distributed.init_process_group(backend='nccl')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.deterministic = True
    
    train_tester = PretrainTrainTester(opt)

    if opt.upload_wandb and opt.local_rank==0:
        run=wandb.init(project="BUTD_DETR")
        run.name = "test_"+run.name
        for k, v in opt.__dict__.items():
            setattr(wandb.config,k,v)

    ckpt_path = train_tester.main(opt)
    
    