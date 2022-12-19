
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

from six.moves import cPickle
import numpy as np
import torch
# import os.path as osp
# import sys
# current = osp.dirname(osp.abspath(__file__))
# root =osp.dirname(current)
# os.chdir(root)
# sys.path.append(root)
# sys.path.append(current)
# sys.path.append(osp.join(root,'my_script'))
# sys.path.append(osp.join(root,'src'))
from src.scannet_classes import  VIEW_DEP_RELS
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import *
from loguru import logger 



NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132



class JointSemiSupervisetDataset(Joint3DDataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, dataset_dict={'sr3d': 1, 'scannet': 10},
                 test_dataset='sr3d',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 butd=False, butd_gt=False, butd_cls=False, augment_det=False):
        """Initialize dataset (here for ReferIt3D utterances)."""
        
        super().__init__(dataset_dict,test_dataset,split, overfit,
                 data_path,use_color, use_height, use_multiview,
                 detect_intermediate,butd, butd_gt, butd_cls, augment_det)
 


    def _get_pc(self, anno, scan):
        """Return a point cloud representation of current scene."""
        scan_id = anno['scan_id']
        rel_name = "none"
        if anno['dataset'].startswith('sr3d'):
            rel_name = self._find_rel(anno['utterance'])

        # a. Color
        color = None
        if self.use_color:
            color = scan.color - self.mean_rgb

        # b .Height
        height = None
        if self.use_height:
            floor_height = np.percentile(scan.pc[:, 2], 0.99)
            height = np.expand_dims(scan.pc[:, 2] - floor_height, 1)

        # c. Multi-view 2d features
        multiview_data = None
        if self.use_multiview:
            multiview_data = self._load_multiview(scan_id)

        # d. Augmentations
        #!+========
        origin_pc = np.concatenate([scan.pc.copy(), color.copy()], 1)
        #!+========

        augmentations = {}
        if self.split == 'train' and self.augment:
            rotate_natural = (
                anno['dataset'] in ('nr3d', 'scanrefer')
                and self._augment_nr3d(anno['utterance'])
            )
            rotate_sr3d = (
                anno['dataset'].startswith('sr3d')
                and rel_name not in VIEW_DEP_RELS
            )
            rotate_else = anno['dataset'] == 'scannet'
            rotate = rotate_sr3d or rotate_natural or rotate_else
            pc, color, augmentations = self._augment(scan.pc, color, rotate)
            scan.pc = pc

        # e. Concatenate representations
        point_cloud = scan.pc
        if color is not None:
            point_cloud = np.concatenate((point_cloud, color), 1)
        if height is not None:
            point_cloud = np.concatenate([point_cloud, height], 1)
        if multiview_data is not None:
            point_cloud = np.concatenate([point_cloud, multiview_data], 1)

        return point_cloud, augmentations, scan.color,origin_pc


    
    '''
    description:  对detector 的bbox进行数据增强 ,#! 没有加入噪声
    param {*} self
    param {*} all_detected_bboxes
    param {*} augmentations
    return {*}
    '''
    def transformation_box(self,boxes,augmentations):
        
        #* do not transformation bbox 
        if len(augmentations.keys()) >0:  #* 不是训练的集的话 这个      augmentations 就是空
            all_det_pts = box2points(boxes).reshape(-1, 3)


            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])  
            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])

            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]

            
            all_det_pts += augmentations['shift']
            all_det_pts *= augmentations['scale']
            boxes = points2box(all_det_pts.reshape(-1, 8, 3))

        return boxes


    '''
    description:  获取当前scan 对应的box
    return {*}
    '''
    def get_current_pc_box(self,scan):
        all_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        #* 这获取的是左上角和右下角, 根据增强后的 pc 计算的box 
        all_bboxes_ = np.stack([ scan.get_object_bbox(k).reshape(-1) for k in range(len(scan.three_d_objects)) ])
        # cx, cy, cz, w, h, d
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)
        all_bboxes[:len(all_bboxes_)] = all_bboxes_
        return all_bboxes




    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        # Read annotation
        anno = self.annos[index]
        scan = self.scans[anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)


        
        origin_box = self.get_current_pc_box(scan)


        
        # Populate anno (used only for scannet)
        self.random_utt = False
        if anno['dataset'] == 'scannet':
            self.random_utt = self.joint_det and np.random.random() > 0.5
            sampled_classes = self._sample_classes(anno['scan_id'])
            utterance = self._create_scannet_utterance(sampled_classes)
            
            # Target ids
            if not self.random_utt:  # detection18 phrase
                anno['target_id'] = np.where(np.array([
                    self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] in DC18.nyu40id2class
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()
            else:
                anno['target_id'] = np.where(np.array([
                    self.label_map[
                        scan.get_object_instance_label(ind)
                    ] in DC.nyu40id2class
                    and
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]] in sampled_classes
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()


            # Target names
            if not self.random_utt:
                anno['target'] = [
                    DC18.class2type[DC18.nyu40id2class[self.label_map18[
                        scan.get_object_instance_label(ind)
                    ]]]
                    if self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] != 39
                    else 'other furniture'
                    for ind in anno['target_id']
                ]
            else:
                anno['target'] = [
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]]
                    for ind in anno['target_id']
                ]
            anno['utterance'] = utterance

        # Point cloud representation#* point_cloud == [x,y,z,r,g,b], 50000 points 
        point_cloud, augmentations, og_color ,origin_pc= self._get_pc(anno, scan)


        #!+========================================================
        #* 用场景原始color
        if self.overfit:
            point_cloud = np.copy(np.concatenate([point_cloud[:,:3],og_color],axis=-1) )
            origin_pc =  np.copy(np.concatenate([origin_pc[:,:3],og_color],axis=-1) )
        #!+======================================================== 



        # "Target" boxes: append anchors if they're to be detected
        gt_bboxes, box_label_mask, point_instance_label = \
            self._get_target_boxes(anno, scan)

        # Positive map for soft-token and contrastive losses
        tokens_positive, positive_map = self._get_token_positive_map(anno)

        #* Scene gt boxes, 
        (
            class_ids, all_bboxes, all_bbox_label_mask
        ) = self._get_scene_objects(scan)

        #* Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)

        #!===================
        #* wrong
        # teacher_box = all_bboxes.copy()
        # teacher_box = self.transformation_box(teacher_box,augmentations)
        #* right 
        teacher_box = origin_box

        #!===================

        # Assume a perfect object detector 
        if self.butd_gt:
            all_detected_bboxes = all_bboxes #* 使用groundtruth bbox
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids = class_ids

        # Assume a perfect object proposal stage
        if self.butd_cls:
            all_detected_bboxes = all_bboxes #? 那么  这个detected box 和 auged  pc 能对应上吗? 
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids = np.zeros((len(all_bboxes,)))
            classes = np.array(self.cls_results[anno['scan_id']])
            detected_class_ids[all_bbox_label_mask] = classes[classes > -1]


        # Visualize for debugging
        if self.visualize and anno['dataset'].startswith('sr3d'):
            self._visualize_scene(anno, point_cloud, og_color, all_bboxes)


        # Return
        _labels = np.zeros(MAX_NUM_OBJ)
        if not isinstance(anno['target_id'], int) and not self.random_utt:
            _labels[:len(anno['target_id'])] = np.array([
                DC18.nyu40id2class[self.label_map18[
                    scan.get_object_instance_label(ind)
                ]]
                for ind in anno['target_id']
            ])



       

            

        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': _labels.astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
        }
        
        #* the anno['dataset']=='scanrefer' actually do not need 
        if 'ann_id' in anno.keys() and anno['dataset']=='scanrefer':
            #  anno['ann_id']:
            ret_dict.update({
                'ann_id':anno['ann_id']
            })
        # elif anno['dataset']=='scannet' :
        else:
            #* scannet , because joint_det is added 
            ret_dict.update({
                'ann_id':'-1'
            })



            

        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "tokens_positive": tokens_positive.astype(np.int64),
            "positive_map": positive_map.astype(np.float32),
            "relation": (
                self._find_rel(anno['utterance'])
                if anno['dataset'].startswith('sr3d')
                else "none"
            ),
            "target_name": scan.get_object_instance_label(
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "target_id": (
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "point_instance_label": point_instance_label.astype(np.int64),
            "all_bboxes": all_bboxes.astype(np.float32),
            "all_bbox_label_mask": all_bbox_label_mask.astype(np.bool8),
            "all_class_ids": class_ids.astype(np.int64),
            "distractor_ids": np.array(
                anno['distractor_ids']
                + [-1] * (32 - len(anno['distractor_ids']))
            ).astype(int),
            "anchor_ids": np.array(
                anno['anchor_ids']
                + [-1] * (32 - len(anno['anchor_ids']))
            ).astype(int),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            "target_cid": (
                class_ids[anno['target_id']]
                if isinstance(anno['target_id'], int)
                else class_ids[anno['target_id'][0]]
            ),
            "pc_before_aug":origin_pc.astype(np.float32),
            "teacher_box":teacher_box.astype(np.float32),
            "augmentations":augmentations,
            "supervised_mask":np.array(1).astype(np.int64)

        })
        return ret_dict





if __name__=="__main__":

    train_dataset = JointSemiSupervisetDataset(
        # dataset_dict={'sr3d':1,'scannet':10},
        dataset_dict={'scannet':10},
        test_dataset='sr3d', #? only test set need ? 
        split='train',
        use_color=True, use_height=False,
        overfit=True,
        data_path='datasets/',
        detect_intermediate=True,#? 
        use_multiview=False, #? 
        butd=False,
        butd_gt=False,
        butd_cls=True,
        augment_det=False
    )

    demo =  train_dataset.__getitem__(0)

    print(demo)
    



    