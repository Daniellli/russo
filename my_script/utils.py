'''
Author: xushaocong
Date: 2022-10-02 20:04:19
LastEditTime: 2022-10-04 17:18:33
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/my_script/utils.py
email: xushaocong@stu.xmu.edu.cn
'''


import os

import os.path as osp
import numpy as np

from loguru import logger 

def make_dirs(path):

    if  not osp.exists(path):
        os.makedirs(path)
        


'''
description:  read txt file 
param {*} path
return {*}
'''
def readtxt(path):
    data = None
    with open(path,'r') as f :
        data = f.read()
    return data




'''
description:  read the meta data of SR3D 
param {*} path : val == test  in SR3D 
return {*}
'''
def read_refer_it_3D_txt(path = "/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/sr3d_test_scans.txt"):
    data  = readtxt(path)
    data = data[1:-1].split(',')
    data = [x.replace('"',"").strip() for x in data]
    logger.info(f"scene number : {len(data)}")
    return data




def generate_SR3D_labeled_scene_txt(labeled_ratio):
    
    split='train'

    with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
        scan_ids = set(eval(f.read()))

    num_scans = len(scan_ids)
    logger.info(f"read {num_scans} scenes ") 
    num_labeled_scans = int(num_scans*labeled_ratio)

    choices = np.random.choice(num_scans, num_labeled_scans, replace=False)#* 从num_scans 挑选num_labeled_scans 个场景 出来 
    labeled_scan_names = list(np.array(list(scan_ids))[choices])
    
    with open(os.path.join('data/meta_data/sr3d_{}_{}.txt'.format(split,labeled_ratio)), 'w') as f:
        f.write('\n'.join(labeled_scan_names))
    
    logger.info('\tSelected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names),len(scan_ids )- len(labeled_scan_names)))

    




if __name__ == "__main__":
    # read_refer_it_3D_txt()
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/sr3d_train_scans.txt")
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/nr3d_test_scans.txt")
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/nr3d_train_scans.txt")
    
    #* 生成labeled datasets for SR3D
    #  
    for x in np.linspace(0.1,0.9,9):
        generate_SR3D_labeled_scene_txt(round(x,1))

