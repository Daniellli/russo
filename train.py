'''
Author: xushaocong
Date: 2022-10-04 19:55:17
LastEditTime: 2022-10-08 00:24:22
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/train.py
email: xushaocong@stu.xmu.edu.cn
'''
'''
Author: xushaocong
Date: 2022-10-03 22:00:15
LastEditTime: 2022-10-04 17:28:33
LastEditors: xushaocong
Description:  修改get_datasets , 换成可以添加使用数据集比例的dataloader
FilePath: /butd_detr/train_dist_mod_2.py
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

from main_utils import BaseTrainTester
from data.model_util_scannet import ScannetDatasetConfig
# from src.joint_det_dataset import Joint3DDataset
from src.sr3d_dataset import SR3DDataset

from src.sr3d_labeled_dataset import SR3DLabeledDataset
from src.sr3d_unlabeled_dataset import SR3DUnlabeledDataset



from src.grounding_evaluator import GroundingEvaluator, GroundingGTEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths



from IPython import embed
import ipdb
st = ipdb.set_trace
import scipy.io as scio
import sys 

from utils.box_util import get_3d_box
import json
import wandb
from loguru import logger 
from my_script.utils import * 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



import os.path as osp
import time


import argparse
from torch.nn.parallel import DistributedDataParallel
from main_utils import save_checkpoint,load_checkpoint,get_scheduler

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random 

from my_script.consistant_loss import get_consistency_loss


from models import HungarianMatcher, SetCriterion, compute_hungarian_loss,compute_labeled_hungarian_loss
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
    parser.add_argument('--batch_size', type=str, default="2,8",
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

    parser.add_argument('--consistency_weight', type=float, default=1.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--labeled_ratio', default=0.2, type=float,help=' labeled datasets ratio ')
    parser.add_argument('--rampup_length', type=float, default=None, help='rampup_length')

    


    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train

    return args


class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

        if args.eval and args.local_rank == 0 : #* for debug ? 
            self.vis_save_path = osp.join(args.log_dir,"vis_results",time.strftime("%Y:%m:%d",time.gmtime(time.time()))+"_"+str(int(time.time())))
            os.makedirs(self.vis_save_path)

        self.DEBUG = args.save_input_output
            
            
        

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

        labeled_ratio = 0.2
        
        logger.info(f"labeled_ratio:{labeled_ratio}")
        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        labeled_dataset = SR3DLabeledDataset(
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
        
        unlabeled_dataset = SR3DUnlabeledDataset(
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



        
        test_dataset = SR3DDataset(
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
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19


        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target, #? 
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls, #* 是否使用gt来负责这个visual grounding 而不是 detected bbox 
            pointnet_ckpt=args.pp_checkpoint,  #* pretrained model
            self_attend=args.self_attend
        )

        return model


    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            "det_boxes": batch_data['all_detected_boxes'],
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],
            "det_class_ids": batch_data['all_detected_class_ids']
        }


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
    description:  评估代码,    
    return {*}
    '''
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        if args.test_dataset == 'scannet':
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model,
                criterion, set_criterion, args
            )
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)


        
        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes = ['last_']
            prefixes.append('proposal_')
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
        
        #?
        if args.butd_cls:
            evaluator = GroundingGTEvaluator(prefixes=prefixes)
        else:
            evaluator = GroundingEvaluator(
                only_root=True, thresholds=[0.25, 0.5],
                topks=[1, 5, 10], prefixes=prefixes
            )
        # Main eval branch
        # DEBUG=True
        pred_bboxes = []
        for batch_idx, batch_data in enumerate(test_loader):#* the length of batch data == 26 , 
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            #!==== generate result for upload evaluation server , scanrefer ===============================
            SAVE_RES =  False
            if SAVE_RES: 
                #* end_points['last_sem_cls_scores']  : [B,query_num,distribution for tokens(256)]  
                #* 1. 对 分布取softmax 
                #* 2. 取最大值的index, 如果最大值的索引是255 则 需要 
                #* 3. 对 分布取softmax 
                prefix="last_"
                
                query_dist_map = end_points[f'{prefix}sem_cls_scores'].softmax(-1) #* 
                objectness_preds_batch = torch.argmax(query_dist_map, 2).long() #* 等于255 应该是没有匹配到文本token的, 如paper解释的一样
                pred_masks = (objectness_preds_batch !=255).float()
                
                # end_points['utterances']
                # pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1) # (B,)
                
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

            #!=================================================================================
            if evaluator is not None:
                for prefix in prefixes:
                    evaluator.evaluate(end_points, prefix)
        evaluator.synchronize_between_processes()
        #!===================
        #* dump for upload evaluation server 
        logger.info("dumping...")
        pred_path = os.path.join(args.log_dir, "pred.json")
        with open(pred_path, "w") as f:
            json.dump(pred_bboxes, f, indent=4)
        logger.info("done!")
        
        ans = None
        #!===================
        if dist.get_rank() == 0:
            if evaluator is not None:
                evaluator.print_stats()
                #!===================
                ans = {}
                if args.butd_cls: #* 给定 GT, 进行分类 
                    prefix ='last_' #* last layer 
                    mode ='bbf'  #* Box given span (contrastive)
                    ans['Acc'] = evaluator.dets[(prefix, mode)] / evaluator.gts[(prefix, mode)]

                else:
                    prefix ='last_' #* last layer 
                    mode ='bbf'  #* Box given span (contrastive)
                    topk=1
                    for t in evaluator.thresholds:
                        ans[f'Acc@{t}-top1'] = evaluator.dets[(prefix, t, topk, mode)]/max(evaluator.gts[(prefix, t, topk, mode)], 1)
                #!===================

        return ans

    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader,
                               model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes += [
                f'{i}head_' for i in range(args.num_decoder_layers - 1)
            ]
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes = ['last_']
        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in args.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in args.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])
        for batch_idx, batch_data in enumerate(test_loader):

            #*  all_utterances = [data['utterances'] for data in test_loader]  #* a test for utterances 
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            #* contrast
            proj_tokens = end_points['proj_tokens']  #* (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  #* (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  #* (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores
            #* end contrast
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19]
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls

            #* Parse predictions
            #* for prefix in prefixes:
            #* 最后一部分是 6个decoder layer, 最后一个的输出是前缀是last_ 
            prefix = 'last_'
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)
            


        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )

        return None

    
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

        total_iteration=max(len(labeled_loader),len(unlabeled_loader))
    
        logger.info(f"total_iteration == {total_iteration}")
        consistency_weight = self.get_current_consistency_weight(args.consistency_weight ,epoch,args)
        logger.info(f"consistency_weight  : {consistency_weight}")

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

            center_consistency_loss = end_points['center_consistency_loss']
            soft_token_consistency_loss = end_points['soft_token_consistency_loss']
            size_consistency_loss = end_points['size_consistency_loss']
            consistent_loss = (soft_token_consistency_loss+center_consistency_loss+size_consistency_loss)* consistency_weight


            #* total loss
            if consistent_loss is not None:
                total_loss = loss+consistent_loss
            else:
                total_loss = loss

            

            #!===================
            if args.upload_wandb and args.local_rank==0:
                
                wandb.log({"student_supervised_loss":loss.clone().detach().item(),
                            "center_consistency_loss":center_consistency_loss.clone().detach().item(),
                            "soft_token_consistency_loss":soft_token_consistency_loss.clone().detach().item(),
                            "size_consistency_loss":size_consistency_loss.clone().detach().item(),
                            "consistent_loss":consistent_loss.clone().detach().item(),
                            "total_loss":total_loss.clone().detach().item(),
                        })
            
            optimizer.zero_grad()
            total_loss.backward()
            #!===================
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
            alpha = 0.999
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

                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0


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
        criterion = compute_labeled_hungarian_loss
        

        return criterion, set_criterion


    def main(self, args):

        
        #!======================= 避免数据跑到其他卡上
        torch.cuda.set_device(args.local_rank)
        logger.info(f"args.local_rank == {args.local_rank}")
        #!=======================

        """Run main training/testing pipeline."""
        # Get loaders
        labeled_loader,unlabeled_loader, test_loader = self.get_loaders(args)
        logger.info(f"length of  labeled dataset: {len(labeled_loader.dataset)} \n  length of  unlabeled dataset: {len(unlabeled_loader.dataset)} \n length of testing dataset: {len(test_loader.dataset)}")
        
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
        scheduler = get_scheduler(optimizer, len(labeled_loader)+len(unlabeled_loader), args)#* 第二个参数是一个epoch需要iteration 多少次 

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda(args.local_rank)
            ema_model = ema_model.cuda(args.local_rank)


        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=True  # , find_unused_parameters=True
        )



        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            load_checkpoint(args, ema_model, optimizer, scheduler,distributed2common=True)


        # Training loop
        #!===============================
        best_performce = 0
        save_dir = osp.join(args.log_dir,'performance.txt')

        ema_best_performce = 0
        ema_save_dir = osp.join(args.log_dir,'ema_performance.txt')

        if osp.exists(save_dir):
            os.remove(save_dir)
        #!===============================
        
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
                # else :
                #     embed()
                
                
                ema_performance = self.evaluate_one_epoch(
                    epoch, test_loader,
                    ema_model, criterion, set_criterion, args
                )
                if ema_performance is not None :
                    logger.info(','.join(['teacher_%s:%.04f'%(k,round(v,4)) for k,v in ema_performance.items()]))
                # else :#* check what cause None ?
                #     embed()
                

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

                    if ema_performance is not None :
                        wandb.log({'teacher_%s'%(k):round(v,4) for k,v in ema_performance.items()})
                        is_best,new_performance = save_res(ema_save_dir,epoch,ema_performance,ema_best_performce)
                        if is_best:
                            ema_best_performce =  new_performance
                            save_checkpoint(args, epoch, ema_model, optimizer, scheduler ,is_best=True,prefix='teacher_')     
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



def save_res(save_dir,epoch,performance,best_performce):
    is_best=False
    with open(save_dir, 'a+') as f :
        f.write( f"epoch:{epoch},"+','.join(["%s:%.4f"%(k,v) for k,v in performance.items()])+"\n")

    acc_key = list(performance.keys())[0]
    if performance is not None and performance[acc_key] > best_performce:
        is_best=True
    return is_best,performance[acc_key]


    
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
    
    # torch.cuda.set_device(opt.local_rank)
    train_tester = TrainTester(opt)

    if opt.upload_wandb and opt.local_rank==0:
        run=wandb.init(project="BUTD_DETR")
        run.name = "test_"+run.name
        for k, v in opt.__dict__.items():
            setattr(wandb.config,k,v)

    ckpt_path = train_tester.main(opt)
    
    