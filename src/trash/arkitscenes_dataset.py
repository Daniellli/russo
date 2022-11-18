import os
import random
import sys
import json

import IPython
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


#!+=====================
from data.model_util_scannet import ScannetDatasetConfig
import my_script.pc_utils as pc_util
from  my_script.pc_utils  import dump_pc
from data.model_util_scannet import rotate_aligned_boxes
import src.arkitscenes_utils as arkitscenes_utils
from loguru import logger 
import os.path as osp 
#!+=====================

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MAX_NUM_QUAD = 32
NUM_PROPOSAL = 256
NUM_QUAD_PROPOSAL = 256

type2class = {
    "cabinet": 0, "refrigerator": 12, "shelf": 7, "stove": -1, "bed": 1, # 0..5
    "sink": 15, "washer": -1, "toilet": 14, "bathtub": 16, "oven": -1, # 5..10
    "dishwasher": -1, "fireplace": -1, "stool": -1, "chair": 2, "table": 4, # 10..15
    "tv_monitor": -1, "sofa": 3, # 15..17
}

def is_valid_mapping_name(data_root,mapping_name):
    mapping_file = os.path.join(data_root, "data", "annotations", f"{mapping_name}.json")
    if os.stat(mapping_file).st_size < 60:
        return False
    return True


class ARKitSceneDataset(Dataset):
    '''
    description: 
    return {*}
    '''
    def __init__(self, split_set='train', num_points=40000,data_root='datasets/ARKitScenes',
        augment=False, start_proportion=0.0, end_proportion=1.0,):

        self.data_root = data_root
        self.data_path = osp.join(data_root,'dataset')
        
        assert split_set in ['train', 'valid']
        self.split_set = split_set
        
        with open(os.path.join(self.data_path, f"{split_set}_filtered.txt"), 'r') as f:
            split_filenames = f.read().strip().split('\n')
       
        if split_set == "train":
            self.data_path = os.path.join(self.data_path, "3dod/Training")
        else:
            self.data_path = os.path.join(self.data_path, "3dod/Validation")
            self.valid_mapping = {line.split(",")[0]: line.split(",")[1] \
                                  for line in open(os.path.join(data_root, 'data', "file.txt")).read().strip().split("\n")}
        
        self.scan_names = sorted(split_filenames)
        bak_scan_names = self.scan_names
        
        self.start_idx = int(len(self.scan_names) * start_proportion)
        self.end_idx = int(len(self.scan_names) * end_proportion)
        self.scan_names = self.scan_names[self.start_idx:self.end_idx]

        # TODO: filter out unlabelled layout in valid set
        #!+====================================================================
        if self.split_set == "valid":
            self.scan_names = [scan_name for scan_name in self.scan_names if is_valid_mapping_name(self.data_root,self.valid_mapping[scan_name])]
        #!+====================================================================



        if len(self.scan_names) == 0:
            self.scan_names = [bak_scan_names[-1], ]
        
        print(f"Find {len(self.scan_names)} in ARKitScene dataset!")
        
        self.num_points = num_points
        self.augment = augment
    
    
    def __len__(self):
        return len(self.scan_names)
    
    def __getitem__(self, idx, **kwargs):

        scan_name = self.scan_names[idx]
        # scan_dir = os.path.join(self.data_path, scan_name, f"{scan_name}_offline_prepared_data_new3")
        scan_dir = os.path.join(self.data_path, scan_name, f"{scan_name}_offline_prepared_data")
        mesh_vertices   = np.load(os.path.join(scan_dir, f'{scan_name}_data',f"{scan_name}_pc.npy"))

        # color= np.load(os.path.join(scan_dir,f"{scan_name}_color.npy"))
        # mesh_vertices = np.concatenate([mesh_vertices,color],axis=-1)
        instance_bboxes = np.load(os.path.join(scan_dir, f'{scan_name}_label', f"{scan_name}_bbox.npy"), allow_pickle=True).item()
    
        # Prepare label containers
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))


        
        # TODO: OBB-Guided Scene Axis-Alignment
        #!+==========================================================================================================
        mesh_vertices_prime = mesh_vertices  
        logger.info(f" vertex number : {mesh_vertices.shape}")

        def transform_box():
            #* 计算旋转矩阵, 用于将pc 转换成跟pc 坐标轴平行
            #* 最后一维度是theta , 也就是box翻转的角度 ,  对90取余, 然后1/2 中位数, 用于表征需要这个pc scene 旋转多少度来与坐标轴对其
            angle = np.percentile(instance_bboxes['bboxes'][..., -1] % (np.pi / 2), 50)
            rot_mat = pc_util.rotz(angle)
            #* 计算将pc移动到原点的偏移矩阵
            #* 取z轴  的15-85 这部分的pc 统计其xy 轴中位数, 将其移动到原点
              
            z_filter_L = np.percentile(mesh_vertices_prime[..., 2], 15)
            z_filter_H = np.percentile(mesh_vertices_prime[..., 2], 85)
            filter_mask = (mesh_vertices_prime[..., 2] >= z_filter_L) & (mesh_vertices_prime[..., 2] <= z_filter_H)
            x_base = np.percentile(mesh_vertices_prime[filter_mask, 0], 50)
            y_base = np.percentile(mesh_vertices_prime[filter_mask, 1], 50)
            z_base = np.percentile(mesh_vertices_prime[..., 2], 5)
            offset = np.array([x_base, y_base, z_base])
            
            #*  transform box  accoring to the rot_mat and offset
            #* mesh_vertices_prime = mesh_vertices_prime - offset
            instance_bboxes['bboxes'][..., :3] = np.dot(instance_bboxes['bboxes'][..., :3], np.transpose(rot_mat))
            instance_bboxes['bboxes'][..., :3] = instance_bboxes['bboxes'][..., :3] - offset
            instance_bboxes['bboxes'][..., 6] -= angle
            instance_bboxes['bboxes'][..., 6] %= 2 * np.pi

            reverse_mask = ((np.pi / 4 <= instance_bboxes['bboxes'][..., 6]) & (instance_bboxes['bboxes'][..., 6] <= np.pi / 4 * 3)) | \
                ((np.pi / 4 * 5 <= instance_bboxes['bboxes'][..., 6]) & (instance_bboxes['bboxes'][..., 6] <= np.pi / 4 * 7))

            dx = np.copy(instance_bboxes['bboxes'][..., 3])
            dy = np.copy(instance_bboxes['bboxes'][..., 4])
            instance_bboxes['bboxes'][..., 3] = dy * reverse_mask + dx * (1-reverse_mask)
            instance_bboxes['bboxes'][..., 4] = dx * reverse_mask + dy * (1-reverse_mask)
        #!+==========================================================================================================


        #* process the box: box number , box class label ,   box  mask 
        bbox_num = min(instance_bboxes['bboxes'].shape[0], MAX_NUM_OBJ)
        rot_mat = pc_util.rotz( - instance_bboxes['bboxes'][:, 6])
        instance_bboxes['bboxes'][..., :3] = np.dot(instance_bboxes['bboxes'][..., :3], np.transpose(rot_mat))
        
        target_bboxes[0:bbox_num, :] = instance_bboxes['bboxes'][:, 0:6]

        
        target_bboxes_mask[0:bbox_num] = 1
        for n_bbox in range(bbox_num):
            str_type = instance_bboxes['types'][n_bbox]
            target_bboxes_semcls[n_bbox] = type2class[str_type]
        num_gt_boxes = np.zeros((NUM_PROPOSAL)) + bbox_num



        #* downsampling 
        point_cloud, choices = pc_util.random_sampling(mesh_vertices_prime,
            self.num_points, return_choices=True)    
        
        ema_point_clouds, ema_choices = pc_util.random_sampling(mesh_vertices_prime,
            self.num_points, return_choices=True)
        
        
        #* data augmentation
        #todo : no augment for pc color 
        
        flip_YZ_XZ = np.array([False, False])
        rot_mat = np.identity(3)
        scale_ratio = np.array(1.0)

        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_YZ_XZ[0] = True
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
                
            
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                flip_YZ_XZ[0] = False
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
                
            
            # Rotation along up-axis / Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36
            rot_angle += random.choice([0, 1, 2, 3]) * np.pi / 2
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            
            #? rotate_aligned_boxes 是干嘛? 
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)
            
            # Augment point cloud scale: 0.85x - 1.15x
            # Assertion: Augmenting scales of point clouds does not change normals
            scale_ratio = np.random.random() * 0.3 + 0.85
            point_cloud[:, 0:3] *= scale_ratio
            target_bboxes[:, 0:3] *= scale_ratio
            target_bboxes[:, 3:6] *= scale_ratio


        
        ret_dict = {
            # Basic
            "scan_ids": scan_name,
            "point_clouds": point_cloud.astype(np.float32),

            # Data augmentation
            "ema_point_clouds": ema_point_clouds.astype(np.float32),
            "flip_x_axis": np.array(flip_YZ_XZ)[..., 0].astype(np.int64),
            "flip_y_axis": np.array(flip_YZ_XZ)[..., 1].astype(np.int64),
            "rot_mat": rot_mat.astype(np.float32),
            "scale": np.array(scale_ratio).astype(np.float32),
            # Label
            "all_bboxes": target_bboxes.astype(np.float32),
            "all_bbox_label_mask":target_bboxes_mask.astype(np.bool8),
            "all_class_ids": target_bboxes_semcls.astype(np.int64),
            #!===================
            "is_view_dep": False,
            "is_hard": False,
            "is_unique": True,
            #!===================

            "num_gt_boxes": num_gt_boxes.astype(np.int64),
            "supervised_mask":np.array(0).astype(np.int64)
        }


        return ret_dict


if __name__ == "__main__":
    dset = ARKitSceneDataset(split_set="train")
    from tqdm import tqdm
    for example in tqdm(dset):
        pc = example['point_clouds']
        normal = example['vertex_normals']
        
        # center = example['center_label']
        # size = example['size_label']
        # scan_name = example['scan_name']
        # box = np.concatenate([center, size], axis=1)
        # os.makedirs("../dump/ARKitDump/", exist_ok=True)
        # dump_pc(pc, f"../dump/ARKitDump/{scan_name}_pc.txt", normal)
        # pc_util.write_bbox(box, f"../dump/ARKitDump/{scan_name}_box.ply")
        # print(pc.shape[0])
        # assert pc.shape[0] > 40000, f"{example['scan_name']}"

