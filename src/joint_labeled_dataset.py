
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

from utils.pc_utils import *

NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132
from os.path import join, split,exists,isdir,isfile

class JointLabeledDataset(Joint3DDataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, dataset_dict={'sr3d': 1, 'scannet': 10},
                 test_dataset='sr3d',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 butd=False, butd_gt=False, butd_cls=False, 
                 augment_det=False,labeled_ratio=None):
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
        """"
        # Fix missing target reference 
        for example: 
            target label : kitchen counter , which not appear in the sentence:
            'it is a light gray counter . it sit on top of wooden cabinets that go along one side of the kitchen . it is on the same side of the kitchen as the refrigerator .' 

            so after fixing, the sentence become :

            'it is a light gray kitchen counter . it sit on top of wooden cabinets that go along one side of the kitchen . it is on the same side of the kitchen as the refrigerator .'

        
        """
        
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

            """
                distractor: the obj has same label with target.  

                #* the max distractor number is 32... 
            """
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
            """"
                np.array(scene2obj[anno['scan_id']]): return all label of object existing in the scene 
                labels[anno['target_id']]: the label of refered target  
            """
            anno['unique'] = (
                np.array(scene2obj[anno['scan_id']])
                == labels[anno['target_id']]
            ).sum() == 1

        #!+=====================================================1. for plot qualitive result ===============================================
        # pick_up_list = np.loadtxt('logs/find_by05iou_list.txt',dtype=np.str0)
        # new_annos = []
        # early_out_idx = 100 
        # for idx,ann in enumerate(annos):
        #     scene_name  = "__".join([ann['scan_id'],str(ann['target_id']), ann['ann_id']])
        #     if scene_name in pick_up_list:
        #         new_annos.append(ann)

        #     if idx == early_out_idx:
        #         break
                
        # return new_annos
        #!+==================================================================================================================================
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
            class_ids, all_bboxes, all_bbox_label_mask,scene_objs_point_instance_label
        ) = self._get_scene_objects(scan)





        #* Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)


        #* wrong
        # teacher_box = all_bboxes.copy()
        # teacher_box = self.transformation_box(teacher_box,augmentations)
        #* right 
        teacher_box = origin_box


        """
        write_ply(point_cloud[:,:3], 'logs/debug/ddebug/pc.ply')
        write_ply(point_cloud[scene_objs_point_instance_label>=0][:,:3], 'logs/debug/ddebug/scene_obj.ply')
        """
        
        
        

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
            "scene_objs_point_instance_label":scene_objs_point_instance_label.astype(np.int64),
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


        #!=================================== 2. for plot  qualitative results =================================== 
        # debug = True
        # if debug == True:
        #     scene_name  = "__".join([anno['scan_id'],str(anno['target_id']), anno['ann_id']])
        #     save_path = f"logs/debug/scene/{scene_name}"
        #     make_dirs(save_path)
        #     #* scene 
        #     write_ply_rgb(point_cloud[:,:3],og_color*256,join(save_path,'scene.ply'))

        #     #* utterances 
        #     np.savetxt(join(save_path,'utterances.txt'),[ret_dict['utterances']],fmt='%s')

        #     #* box 
        #     write_bbox(np.concatenate([ret_dict['center_label'],
        #                 ret_dict['size_gts']],axis=-1)[ret_dict['box_label_mask']==1],
        #                 join(save_path,'target_box.ply'))
        #!========================================================================================================= 
        return ret_dict





if __name__=="__main__":

    train_dataset = JointLabeledDataset(
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
    



    