
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
from my_utils.utils import * 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import os.path as osp
import time
from my_utils.utils import parse_semi_supervise_option,save_res


class OmniSuperviseTrainTester(SemiSuperviseTrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)
        


    
    '''
    description:  重写get_unlabeled_dataset function, 很多参数都用不上, 
     但是为了和train.py 公用一个main(), 保留了这些没用的参数
    return {*}
    '''
    def get_unlabeled_dataset(self,data_root,train_dataset_dict,test_datasets,split,use_color,use_height,
                    detect_intermediate,use_multiview,butd,butd_gt,butd_cls,
                    augment_det=False,debug=False,labeled_ratio=None,unlabel_dataset_root=None):
        

        logger.info(f"unlabeld datasets: arkitscenes has been loaded ")
        
        return  UnlabeledARKitSceneDataset(augment=True,
            data_root=unlabel_dataset_root,
            butd_cls=butd_cls)
    





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
    
    