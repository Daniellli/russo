

import os
import os.path as osp 
import random
import sys
import json
import IPython
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from six.moves import cPickle

# current = osp.dirname(osp.abspath(__file__))
# root =osp.dirname(current)
# os.chdir(root)
# sys.path.append(root)
# sys.path.append(current)
# sys.path.append(osp.join(root,'my_script'))



from data.model_util_scannet import ScannetDatasetConfig,rotate_aligned_boxes
from my_utils.utils import print_attr_shape,make_dirs
import my_utils.pc_utils  as pc_util
#* debug 
# import pc_utils  as pc_util



from src.labeled_arkitscenes_dataset import LabeledARKitSceneDataset


from loguru import logger 
import trimesh



from src.joint_det_dataset import rot_x,rot_y,rot_z,box2points,points2box
import torch
from transformers import RobertaTokenizerFast
from IPython import embed



# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)
# sys.path.append(BASE_DIR)

DC = ScannetDatasetConfig()
DC18 = ScannetDatasetConfig(18)

from utils.taxonomy import ARKitDatasetConfig
ARKit_DC=ARKitDatasetConfig()

# MAX_NUM_OBJ = 64
# NUM_PROPOSAL = 256
MAX_NUM_OBJ = 132
NUM_CLASSES = 485




def is_valid_mapping_name(data_root,mapping_name):
    mapping_file = os.path.join(data_root, "data", "annotations", f"{mapping_name}.json")
    if os.stat(mapping_file).st_size < 60:
        return False
    return True



class UnlabeledARKitSceneDataset(LabeledARKitSceneDataset):


    def __init__(self, num_points=50000,data_root='datasets/ARKitScenes',
        augment=False,debug=False,butd_cls = False ):

        super().__init__(
            num_points,data_root,augment,debug,butd_cls)


    def __getitem__(self, idx, **kwargs):
        
        
        scan_name = list(self.annos.keys())[idx]
        anno  = self.annos[scan_name]
        
        #* load pc
        scan_dir = os.path.join(anno['data_path'], scan_name, f"{scan_name}_offline_prepared_data_2")
        mesh_vertices = np.load(os.path.join(scan_dir, f"{scan_name}_data", f"{scan_name}_pc.npy"))
        


       #* num_gt_boxes  : 感觉这个没什么用
        point_clouds,bboxes = self.align_box_pc(mesh_vertices,anno['bboxes'])

        #todo statistic data  
        #*1.  一些如果场景box  只会保留10个作为target 
        #*2.  一些 box label 没有出现在scannet 类别里也不考虑作为 target 
        sampled_classes = self._sample_classes(anno)#* 会删除一下box对应的label 
        utterance = self._create_scannet_utterance(sampled_classes)
        #* reverse all target in the scene 
        #* 对应的box的索引 ,   
        anno['target_id'] = np.where(([idd !=-1  and DC.class2type[idd] in sampled_classes  for idd in  anno['bbox_class_ids_in_scannet']   ])[:MAX_NUM_OBJ])[0].tolist()
        anno['target'] = [anno['bbox_class_in_arkitscenes'][ind]  for ind in anno['target_id'] ]
        anno['utterance'] = utterance
        
                
        point_clouds, augmentations, og_color,origin_pc=self._get_pc(point_clouds)

        #* teacher's things
        teacher_box = np.zeros((MAX_NUM_OBJ, 6))
        teacher_box[:bboxes.shape[0],:] = np.copy(bboxes)
        teacher_pc = origin_pc


        
        #* align box 
        bboxes = self.align_box_to_pc(bboxes,augmentations)

        
        if self.debug:
            # debug_pc = np.concatenate([origin_pc[:,:3],og_color],axis=-1)
            debug_pc = np.concatenate([point_clouds[:,:3],og_color],axis=-1) #* 用augment 之后的数据可视化检查

        

        #* 获取对应的 utterance target 对应的box, 
        #! 少了point_instance_label
        gt_bboxes, box_label_mask = self._get_target_boxes(bboxes,anno)
        tokens_positive, positive_map = self._get_token_positive_map(anno)


        #* scene gt box , already done
        class_ids,all_bboxes,all_bbox_label_mask=self._get_scene_objects(bboxes,anno)
        


        all_detected_bboxes = np.zeros(all_bboxes.shape)
        all_detected_bbox_label_mask = np.zeros(all_bbox_label_mask.shape)
        detected_class_ids = np.zeros((len(all_bboxes,)))
        if self.butd_cls:
            all_detected_bboxes = all_bboxes #? 那么  这个detected box 和 auged  pc 能对应上吗? 
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids[all_bbox_label_mask] = class_ids[class_ids !=0]



        #! 少了  all_detected_bboxes, all_detected_bbox_label_mask, detected_class_ids, detected_logits

        #* sem_cls_label 没有用! 
        ret_dict = {
            # 'box_label_mask': box_label_mask.astype(np.float32),
            # 'center_label': gt_bboxes[:, :3].astype(np.float32),
            # 'sem_cls_label': np.zeros(MAX_NUM_OBJ).astype(np.int64),
            # 'size_gts': gt_bboxes[:, 3:].astype(np.float32),
        }

        

        
        ret_dict.update( {
            # Basic
            "scan_ids": scan_name,
            "point_clouds": point_clouds.astype(np.float32) if not  self.debug  else debug_pc.astype(np.float32), 
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            # "tokens_positive": tokens_positive.astype(np.int64),
            # "positive_map": positive_map.astype(np.float32),
            # "relation": ("none"),
            # "target_name": anno['target'][0],
            # "target_id": (anno['target_id'][0]),

            # "all_bboxes": all_bboxes.astype(np.float32),
            # "all_bbox_label_mask":all_bbox_label_mask.astype(np.bool8),
            # "all_class_ids": class_ids.astype(np.int64),

            #! no detected results: 
            #* for cls task 
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64) ,
            # "all_detected_logits": detected_logits.astype(np.float32) ,
            #*! no point_instance_label  , namely sematic results

            # "distractor_ids": np.array(
            #     anno['distractor_ids'] + [-1] * (32 - len(anno['distractor_ids']))
            # ).astype(int),
            # "anchor_ids": np.array(
            #     anno['anchor_ids'] + [-1] * (32 - len(anno['anchor_ids']))
            # ).astype(int),
            
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            #?  对应scennet calss set 的class ID
            # "target_cid": (anno['bbox_class_ids_in_scannet'][anno['target_id'][0]]),
            "pc_before_aug":teacher_pc.astype(np.float32),
            "teacher_box":teacher_box.astype(np.float32),
            #! no teacher box because no  detected results, so we can only test on det setting 
            "augmentations":augmentations,
            "supervised_mask":np.array(0).astype(np.int64),#*  2 表示有标签 但是没有point_instance_label
        })


        return ret_dict





if __name__ == "__main__":
    dset = UnlabeledARKitSceneDataset( augment=True,data_root='datasets/arkitscenes',butd_cls = True)

    
    from tqdm import tqdm
    i=0
    debug_path = "logs/debug"
    # demo = dset.__getitem__(4)
    

    for example in tqdm(dset):
        target_num = example['box_label_mask'].astype(np.int64).sum()

        logger.info(f"target number == {target_num}")
        if len(example['target_name']) == 0:
            logger.info(example['scan_ids'])
            logger.info(f"target number :{len(example['target_name'])}")
            
        # save_(example,example['scan_ids'],debug_path)
        # print_attr_shape(example)
        # if i == 2:
        #     break
        i+=1
        
