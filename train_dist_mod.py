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
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset,points2box
from src.grounding_evaluator import GroundingEvaluator, GroundingGTEvaluator
# from models import BeaUTyDETR
from models import BeaUTyDETRTKPS
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from my_script.utils import save_for_vis,parse_option
import os.path as osp
import time


class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

        if args.eval and args.local_rank == 0 : #* for debug ? 
            self.vis_save_path = osp.join("vis_results",time.strftime("%Y:%m:%d",time.gmtime(time.time()))+"_"+str(int(time.time())))
            os.makedirs(self.vis_save_path)

            
        

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
                    detect_intermediate,use_multiview,butd,butd_gt,butd_cls,augment_det=False,debug=False):

        return Joint3DDataset(
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
            augment_det=augment_det 
        )
        

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


        # model = BeaUTyDETR(
        #     num_class=num_class,
        #     num_obj_class=485,
        #     input_feature_dim=num_input_channel,
        #     num_queries=args.num_target, #? 
        #     num_decoder_layers=args.num_decoder_layers,
        #     self_position_embedding=args.self_position_embedding,
        #     contrastive_align_loss=args.use_contrastive_align,
        #     butd=args.butd or args.butd_gt or args.butd_cls, #* 是否使用gt来负责这个visual grounding 而不是 detected bbox 
        #     pointnet_ckpt=args.pp_checkpoint,  #* pretrained model
        #     self_attend=args.self_attend
        # )

        model = BeaUTyDETRTKPS(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target, #? 
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls, #* 是否使用gt来负责这个visual grounding 而不是 detected bbox 
            pointnet_ckpt=args.pp_checkpoint,  #* pretrained model
            self_attend=args.self_attend,
            use_tkps=args.use_tkps,
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


    
    '''
    description:  评估并且存储评估结果用于上传ScanRefer server for evaluation   ,
    for_vis: 是否存储可视化需要的数据 
    return {*}
    '''
    @torch.no_grad()
    def evaluate_one_epoch_save_for_eval(self, epoch, test_loader,
                           model, criterion, set_criterion, args,for_vis = False):
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
        
        if args.butd_cls:
            evaluator = GroundingGTEvaluator(prefixes=prefixes)
        else:
            evaluator = GroundingEvaluator(
                only_root=True, thresholds=[0.25, 0.5],
                topks=[1, 5, 10], prefixes=prefixes
            )
        # Main eval branch

        pred_bboxes = []
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': ScannetDatasetConfig(18),
            'hungarian_loss': True
        }

        for batch_idx, batch_data in enumerate(test_loader):#* the length of batch data == 26 , 

            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            #!==== generate result for upload evaluation server , scanrefer ===============================

            
            prefix = 'last_'
            #* return format :  pred_cls, pred_box and conf (0-1)
            #* box 是8个定点
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)

            for idx,batch_res in  enumerate(batch_pred_map_cls):

                score = np.array([x[2] for x in batch_res])
                batch_res=  np.array(batch_res)
                max_idx = np.argmax(score)#* 只取confidence 最大的, 不管是什么哪个target  , 这个对应的是target id , 也就是第几个目标
                obj_id=batch_res[max_idx][:,0] #?
                boxes =batch_res[max_idx][:,1]

                pred_data = {
                    "scene_id": end_points['scan_ids'][idx],
                    "object_id": obj_id,
                    "ann_id": end_points['ann_id'][idx],#* 训练的时候没用到, 所以dataloader没有加载
                    "bbox": boxes.tolist(),
                    "unique_multiple":  end_points["is_unique"][idx].item()==False, #* return true means multiple 
                    "others": 1 if end_points["target_cid"][idx] == 17 else 0
                }

                pred_bboxes.append(pred_data)
                
                #* save for vis , get top k for  vis 
                
                if for_vis:
                    topk = 10
                    max_indexes = np.argsort(score)[-topk:]
                    # score[max_indexes][::-1]#* 倒序的 转 正序的

                    obj_id=batch_res[max_indexes][:,0] #?
                    boxes =batch_res[max_indexes][:,1]
                    boxes = np.array([ box.tolist() for box in boxes])
                    boxes = points2box(boxes)
                    save_for_vis(boxes,batch_data['point_clouds'][idx],self.vis_save_path,end_points['scan_ids'][idx],flag='student')
                    save_for_vis(batch_data['all_bboxes'][idx][batch_data['all_bbox_label_mask'][idx]].cpu().numpy(),batch_data['point_clouds'][idx],self.vis_save_path,end_points['scan_ids'][idx],flag='student2')
                    save_for_vis((torch.cat([batch_data['size_gts'],batch_data['center_label']],axis=-1)[idx][batch_data['box_label_mask'][idx].bool()]).cpu().numpy()
                                    ,batch_data['point_clouds'][idx],self.vis_save_path,end_points['scan_ids'][idx],flag='teacher')


            #!=================================================================================
            if evaluator is not None:
                for prefix in prefixes:
                    evaluator.evaluate(end_points, prefix)
                    
        evaluator.synchronize_between_processes()
        
        #* dump for upload evaluation server ========================================
        logger.info("dumping...")
        pred_path = os.path.join(args.log_dir, "pred.json")
        with open(pred_path, "w") as f:
            json.dump(pred_bboxes, f, indent=4)
        logger.info("done!")
        #*===========================================================================

        ans = None
        if dist.get_rank() == 0:
            if evaluator is not None:
                evaluator.print_stats()
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
            #* contrast : 计算text token and queries 之间的相似度
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
            #* 一般最后一个layer的输出是最好的,  取最后一个结果作为预测结果
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

    if opt.eval:
        train_tester.evaluation(opt)
        exit(0)

    ckpt_path = train_tester.main(opt)
    
    