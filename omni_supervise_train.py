'''
Author: xushaocong
Date: 2022-10-03 22:00:15
LastEditTime: 2022-10-27 23:11:53
LastEditors: xushaocong
Description:  修改get_datasets , 换成可以添加使用数据集比例的dataloader
FilePath: /butd_detr/omni_supervise_train.py
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
import torch
import torch.distributed as dist
from train import SemiSuperviseTrainTester
from src.joint_semi_supervise_dataset import JointSemiSupervisetDataset
from src.unlabeled_arkitscenes_dataset import UnlabeledARKitSceneDataset
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
from my_script.utils import parse_semi_supervise_option,save_res


class OmniSuperviseTrainTester(SemiSuperviseTrainTester):
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
        

        arkitscenes_dataset = UnlabeledARKitSceneDataset(
            augment=True,
            data_root=args.unlabel_dataset_root,
            butd_cls=args.butd_cls)
        
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

        
        return labeled_dataset,arkitscenes_dataset, test_dataset


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
    train_tester = OmniSuperviseTrainTester(opt)

    if opt.upload_wandb and opt.local_rank==0:
        run=wandb.init(project="BUTD_DETR")
        run.name = "test_"+run.name
        for k, v in opt.__dict__.items():
            setattr(wandb.config,k,v)

    ckpt_path = train_tester.main(opt)
    
    