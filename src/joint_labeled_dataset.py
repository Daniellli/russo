
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
from src.joint_semi_supervise_dataset import JointSemiSupervisetDataset

from loguru import logger 

from IPython import embed 


NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132


class JointLabeledDataset(JointSemiSupervisetDataset):
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
        logger.info(f"in JointLabeledDataset, load_scanrefer_annos ")
        _path = self.data_path + 'scanrefer/ScanRefer_filtered'
        split = self.split
        if split in ('val', 'test'):
            split = 'val'

        if split== 'train' and self.labeled_ratio is not None:
            with open(_path + '_%s_%.1f.txt' % (split,self.labeled_ratio)) as f:
                labeld_scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
            scan_ids = set(labeld_scan_ids)
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
                'dataset': 'scanrefer',
                'ann_id':anno['ann_id']

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
            assignment_ids = set(assignment_ids)

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
        
        #* Filter out sentences that do not explicitly mention the target class
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

    