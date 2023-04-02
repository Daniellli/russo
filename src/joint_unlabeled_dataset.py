
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

import csv
from collections import defaultdict
import h5py
import json
import multiprocessing as mp
import os
import random
from six.moves import cPickle


import numpy as np
import torch
from torch.utils.data import Dataset
#!+=============
from transformers import RobertaTokenizerFast
#!+=============

from src.joint_semi_supervise_dataset import JointSemiSupervisetDataset
from IPython import embed
import wandb

from data.model_util_scannet import ScannetDatasetConfig
from data.scannet_utils import read_label_mapping
from src.visual_data_handlers import Scan
from .scannet_classes import REL_ALIASES, VIEW_DEP_RELS

NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132

import os.path as osp
from loguru import logger 


class JointUnlabeledDataset(JointSemiSupervisetDataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, dataset_dict={'sr3d': 1, 'scannet': 10},
                 test_dataset='sr3d',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 butd=False, butd_gt=False, butd_cls=False, augment_det=False,
                 labeled_ratio=None):
        """Initialize dataset (here for ReferIt3D utterances)."""
        
        self.labeled_ratio = labeled_ratio
        

        super().__init__(dataset_dict,test_dataset,split, overfit,
                        data_path,use_color, use_height, use_multiview,
                        detect_intermediate,butd, butd_gt, butd_cls, augment_det)
                



    def load_scanrefer_annos(self):
        """Load annotations of ScanRefer."""
        _path = self.data_path + 'scanrefer/ScanRefer_filtered'
        split = self.split
        if split in ('val', 'test'):
            split = 'val'

        if split== 'train' and self.labeled_ratio is not None:

            with open(_path + '_%s.txt' % split) as f:
                all_scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]


            with open(_path + '_%s_%.1f.txt' % (split,self.labeled_ratio)) as f:
                labeld_scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
            
            scan_ids = list(set(all_scan_ids) - set(labeld_scan_ids))

            
        else :
            with open(_path + '_%s.txt' % split) as f:
                scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]


            
        with open(_path + '_%s.json' % split) as f:
            reader = json.load(f)
        annos = [
            {
                'scan_id': anno['scene_id'],
                'target_id': int(anno['object_id']),
                'distractor_ids': [],
                'utterance': ' '.join(anno['token']),
                'target': ' '.join(str(anno['object_name']).split('_')),
                'anchors': [],
                'anchor_ids': [],
                'dataset': 'scanrefer'
            }
            for anno in reader
            if anno['scene_id'] in scan_ids
        ]
        # Fix missing target reference
        for anno in annos:
            if anno['target'] not in anno['utterance']:
                anno['utterance'] = (
                    ' '.join(anno['utterance'].split(' . ')[0].split()[:-1])
                    + ' ' + anno['target'] + ' . '
                    + ' . '.join(anno['utterance'].split(' . ')[1:])
                )
        # Add distractor info
        scene2obj = defaultdict(list)
        sceneobj2used = defaultdict(list)
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if labels[ind] == labels[anno['target_id']]
                and ind != anno['target_id']
            ][:32]
            if anno['target_id'] not in sceneobj2used[anno['scan_id']]:
                sceneobj2used[anno['scan_id']].append(anno['target_id'])
                scene2obj[anno['scan_id']].append(labels[anno['target_id']])
        # Add unique-multi
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['unique'] = (
                np.array(scene2obj[anno['scan_id']])
                == labels[anno['target_id']]
            ).sum() == 1
        return annos



    '''
    description:  根据assignment id 来加载数据集
    param {*} self
    return {*}
    '''
    def load_nr3d_annos(self):
        """Load annotations of nr3d."""
        split = self.split
        if split == 'val':
            split = 'test'
            

        if split== 'train' and self.labeled_ratio is not None:
            with open(os.path.join('data/meta_data/nr3d_{}_{}.txt'.format(split,self.labeled_ratio)), 'r') as f:
                assignment_ids = f.read().split('\n')

            with open(os.path.join('data/meta_data/nr3d_{}_all_assignmentid.txt'.format(split)), 'r') as f:
                all_assignment_ids = f.read().split('\n')

            assignment_ids =set(all_assignment_ids) -  set(assignment_ids)

            

            with open(self.data_path + 'refer_it_3d/nr3d.csv') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader)
                headers = {header: h for h, header in enumerate(headers)}
                annos = [
                    {
                        'scan_id': line[headers['scan_id']],
                        'target_id': int(line[headers['target_id']]),
                        'target': line[headers['instance_type']],
                        'utterance': line[headers['utterance']],
                        'anchor_ids': [],
                        'anchors': [],
                        'dataset': 'nr3d'
                    }
                    for line in csv_reader
                    #!+==================
                    # if line[headers['scan_id']] in scan_ids
                    if line[headers['assignmentid']] in assignment_ids
                    #!+==================
                    and
                    str(line[headers['mentions_target_class']]).lower() == 'true'
                    and
                    (
                        str(line[headers['correct_guess']]).lower() == 'true'
                        or split != 'test'
                    )
                ]

        else:
            with open('data/meta_data/nr3d_%s_scans.txt' % split) as f:
                scan_ids = set(eval(f.read()))

            with open(self.data_path + 'refer_it_3d/nr3d.csv') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader)
                headers = {header: h for h, header in enumerate(headers)}
                annos = [
                    {
                        'scan_id': line[headers['scan_id']],
                        'target_id': int(line[headers['target_id']]),
                        'target': line[headers['instance_type']],
                        'utterance': line[headers['utterance']],
                        'anchor_ids': [],
                        'anchors': [],
                        'dataset': 'nr3d'
                    }
                    for line in csv_reader
                    if line[headers['scan_id']] in scan_ids
                    and
                    str(line[headers['mentions_target_class']]).lower() == 'true'
                    and
                    (
                        str(line[headers['correct_guess']]).lower() == 'true'
                        or split != 'test'
                    )
                ]

        # Add distractor info
        for anno in annos:
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if self.scans[anno['scan_id']].get_object_instance_label(ind)
                == anno['target']
                and ind != anno['target_id']
            ]

        # Filter out sentences that do not explicitly mention the target class
        annos = [anno for anno in annos if anno['target'] in anno['utterance']]
        return annos



    def load_sr3d_annos(self, dset='sr3d'):
        """Load annotations of sr3d/sr3d+."""
        split = self.split
        if split == 'val':
            split = 'test'
            
        if split== 'train' and self.labeled_ratio is not None:
            with open(os.path.join('data/meta_data/sr3d_{}_{}.txt'.format(split,self.labeled_ratio)), 'r') as f:
                labeled_scenes = f.read().split('\n')
            
            scan_ids = set(labeled_scenes)
            with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
                all_scan_ids = set(eval(f.read()))

            scan_ids = list(all_scan_ids-scan_ids)
        else :
            with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
                scan_ids = set(eval(f.read()))

        # with open(self.data_path + 'refer_it_3d/%s.csv' % dset) as f:
        with open(self.data_path + '/refer_it_3d/%s.csv' % dset) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = [
                {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'distractor_ids': eval(line[headers['distractor_ids']]),
                    'utterance': line[headers['utterance']],
                    'target': line[headers['instance_type']],
                    'anchors': eval(line[headers['anchors_types']]),
                    'anchor_ids': eval(line[headers['anchor_ids']]),
                    'dataset': dset
                }
                for line in csv_reader
                if line[headers['scan_id']] in scan_ids
                and
                str(line[headers['mentions_target_class']]).lower() == 'true'
            ]
        
        return annos




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
        # (
        #     class_ids, all_bboxes, all_bbox_label_mask
        # ) = self._get_scene_objects(scan)

        #* Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)

        #!===================
        # teacher_box = origin_box
        #!===================

        # Assume a perfect object detector 
        # if self.butd_gt:
        #     all_detected_bboxes = all_bboxes #* 使用groundtruth bbox
        #     all_detected_bbox_label_mask = all_bbox_label_mask
        #     detected_class_ids = class_ids

        # Assume a perfect object proposal stage
        # if self.butd_cls:
        #     all_detected_bboxes = all_bboxes #? 那么  这个detected box 和 auged  pc 能对应上吗? 
        #     all_detected_bbox_label_mask = all_bbox_label_mask
        #     detected_class_ids = np.zeros((len(all_bboxes,)))
        #     classes = np.array(self.cls_results[anno['scan_id']])
        #     detected_class_ids[all_bbox_label_mask] = classes[classes > -1]


        # Visualize for debugging
        # if self.visualize and anno['dataset'].startswith('sr3d'):
        #     self._visualize_scene(anno, point_cloud, og_color, all_bboxes)


        # Return
        # _labels = np.zeros(MAX_NUM_OBJ)
        # if not isinstance(anno['target_id'], int) and not self.random_utt:
        #     _labels[:len(anno['target_id'])] = np.array([
        #         DC18.nyu40id2class[self.label_map18[
        #             scan.get_object_instance_label(ind)
        #         ]]
        #         for ind in anno['target_id']
        #     ])
            

        ret_dict = {


        }
        
        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),

            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            "pc_before_aug":origin_pc.astype(np.float32),

            # "teacher_box":teacher_box.astype(np.float32),

            "augmentations":augmentations,
            "supervised_mask":np.array(0).astype(np.int64)

        })
        return ret_dict
