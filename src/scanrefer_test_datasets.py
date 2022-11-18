
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

import csv
from collections import defaultdict
import json
import os
import numpy as np
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_labeled_dataset import JointLabeledDataset




NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132


class ScanReferTestDataset(JointLabeledDataset):
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


        self.scanrefer_benchmark_data= load_json('datasets/scanrefer/benchmark_data.json')

        
        self.labeled_ratio = labeled_ratio     
        super().__init__(dataset_dict,test_dataset,split, overfit,
                 data_path,use_color, use_height, use_multiview,
                 detect_intermediate,butd, butd_gt, butd_cls, augment_det)


        
    def find_benchmark_data(self,obj_id,ann_id):
        for line in self.scanrefer_benchmark_data:
            if line['object_id']  ==  int(obj_id) and line['ann_id']==int(ann_id):
                return line
                
        
    
    def load_scanrefer_annos(self):
        """Load annotations of ScanRefer."""
        _path = self.data_path + 'scanrefer/ScanRefer_filtered'
        split = self.split

        if split== 'train' and self.labeled_ratio is not None:
            with open(_path + '_%s_%.1f.txt' % (split,self.labeled_ratio)) as f:
                labeld_scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
            scan_ids = set(labeld_scan_ids)
        else :
            with open(_path + '_%s.txt' % split) as f:
                scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]


        with open(_path + '_%s.json' % split) as f:
            reader = json.load(f)


        annos = []
        for anno in reader:
            if anno['scene_id'] in scan_ids:
                #* 通过 object ID 和 ann id 可以唯一定位benchmark data的数据

                benchmark_data = self.find_benchmark_data(anno['object_id'],anno['ann_id'])

                annos.append(
                    {
                    'scan_id': anno['scene_id'],
                    'target_id': int(anno['object_id']),
                    'distractor_ids': [],
                    'utterance': ' '.join(anno['token']),
                    'target': ' '.join(str(anno['object_name']).split('_')),
                    'anchors': [],
                    'anchor_ids': [],
                    'dataset': 'scanrefer',
                    'ann_id': anno['ann_id'],
                    'unique_multiple':benchmark_data['unique_multiple'],
                    "object_cat":benchmark_data['object_cat'],   
                    }
                )

        
        # Fix missing target reference
        for anno in annos:
            if anno['target'] not in anno['utterance']:
                anno['utterance'] = (
                    ' '.join(anno['utterance'].split(' . ')[0].split()[:-1])
                    + ' ' + anno['target'] + ' . '
                    + ' . '.join(anno['utterance'].split(' . ')[1:])
                )

        
        return annos






    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        # Read annotation
        anno = self.annos[index]
        scan = self.scans[anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)
        # Point cloud representation#* point_cloud == [x,y,z,r,g,b], 50000 points 
        point_cloud, augmentations, og_color ,origin_pc= self._get_pc(anno, scan)

        #* Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)



        ret_dict ={
            "scan_ids": anno['scan_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            
            "relation": (
                self._find_rel(anno['utterance'])
                if anno['dataset'].startswith('sr3d')
                else "none"
            ),
            #todo check is it right ? 
            "target_name":anno['target'],
            "target_id": (
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),

            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            'unique_multiple':anno['unique_multiple'],
            'ann_id': anno['ann_id'],
            "target_cid": np.array(anno['object_cat']).astype(np.float32),
            "augmentations":augmentations,
            "supervised_mask":np.array(1).astype(np.int64)

        }
        return ret_dict



    




def load_json(path):
    with open(path,'r')as f :
        return json.load(f)
        