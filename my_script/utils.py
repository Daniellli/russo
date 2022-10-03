'''
Author: xushaocong
Date: 2022-10-02 20:04:19
LastEditTime: 2022-10-03 20:42:57
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/my_script/utils.py
email: xushaocong@stu.xmu.edu.cn
'''


import os

import os.path as osp

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

if __name__ == "__main__":
    # read_refer_it_3D_txt()
    # read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/sr3d_train_scans.txt")
    read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/nr3d_test_scans.txt")
    read_refer_it_3D_txt(path="/home/DISCOVER_summer2022/xusc/exp/butd_detr/data/meta_data/nr3d_train_scans.txt")

