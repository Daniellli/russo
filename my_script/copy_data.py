'''
Author: xushaocong
Date: 2022-10-21 11:22:10
LastEditTime: 2022-10-21 11:47:24
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/my_script/copy_data.py
email: xushaocong@stu.xmu.edu.cn
'''

import glob

import os.path as osp


import shutil
from tqdm import tqdm
import os
import multiprocessing as mp

src_path = "/home/DISCOVER_summer2022/xusc/exp/butd_detr/datasets/arkitscenes/dataset/3dod"
tgt_path='/home/DISCOVER_summer2022/xusc/exp/butd_detr/datasets/ARKitScenes/dataset/3dod'

splits = {
    'train':"Training",'valid':"Validation"
}


for split,item in splits.items():
    src_all_sample_pathes  = os.listdir(osp.join(src_path,item))

    for p in tqdm(src_all_sample_pathes):
        src = osp.join(src_path,item,p,f'{p}_offline_prepared_data_2')
        tgt = osp.join(tgt_path,item,p,f'{p}_offline_prepared_data_2')

        if osp.exists(tgt):
            shutil.rmtree(tgt)
            
        

        if osp.exists(src):
            shutil.copytree(src,tgt)
        


