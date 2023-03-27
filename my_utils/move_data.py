


import numpy as np
import os.path as osp
import sys
import os
root = "~/exp/butd_detr"
sys.path.append(root)
from my_utils.utils import make_dirs
import shutil 

from tqdm import tqdm 
class ARKitScene:
    def __init__(self, split,src_path = '~/exp/butd_detr/datasets/arkitscenes/dataset/3dod'):
        
        self.split=split
        self.detected_path ="~/exp/butd_detr/datasets/fcaf3d_arkitscenes/submit"
        
        if split=="train":
            self.split_branch = 'Training'
            
            self.path = osp.join(src_path,self.split_branch)
        else :
            self.split_branch = 'Validation'
            self.path = osp.join(src_path,self.split_branch)

        self.parse_all_scene()

        self.annotation_format= "%s_offline_prepared_data_2"
        self.detected_format= "%s.bin"
        

    def get_split_branch(self):
        return self.split_branch 
        
    def parse_all_scene(self):
        self.all_scene = os.listdir(self.path)

    def get_len(self):
        return len(self.all_scene)
    
    def get_scene_name(self,idx):
        return self.all_scene[idx]


    def get_annotation_file_name(self,idx):
        
        return self.annotation_format%(self.all_scene[idx])


    def get_annotation_path(self,index):
        
        scene_name =self.all_scene[index]

        scene_path = osp.join(self.path, scene_name,self.get_annotation_file_name(index))

        return scene_path


    
    def get_detected_res_path(self,index):
        scene_name =self.all_scene[index]
        return osp.join(self.detected_path,self.detected_format%(scene_name))


        



splits= ['train','valid']

target_dir = "~/exp/butd_detr/datasets/ARKitScenes/dataset/3dod"

src_path = '~/exp/butd_detr/datasets/arkitscenes/dataset/3dod'


invalid_path  = [] 
for split in splits:
    all_valid_scenes = np.loadtxt(osp.join(osp.dirname(src_path),"%s_filtered.txt"%split),delimiter = '\n',dtype=np.str0).tolist()
    
    spliter = ARKitScene(split,src_path =src_path)

    target_path_split_root = osp.join(target_dir,spliter.get_split_branch())
    
    make_dirs(target_path_split_root)
    
    for idx in tqdm(range(spliter.get_len())):
        scene_name = spliter.get_scene_name(idx)
        if scene_name not in all_valid_scenes:
            invalid_path.append(scene_name)
            print(f"{scene_name} is invalid ")
            
            continue

        target_path = osp.join(target_path_split_root,scene_name)
        make_dirs(target_path)
        src_det = spliter.get_detected_res_path(idx)
        src_ann = spliter.get_annotation_path(idx)
        
        
        shutil.copy(src_det,target_path)
        shutil.copytree(src_ann,osp.join(target_path,spliter.get_annotation_file_name(idx)))
    
