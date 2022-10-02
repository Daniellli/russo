'''
Author: xushaocong
Date: 2022-10-02 20:04:19
LastEditTime: 2022-10-02 20:04:20
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/my_script/utils.py
email: xushaocong@stu.xmu.edu.cn
'''


import os

import os.path as osp


def make_dirs(path):

    if  not osp.exists(path):
        os.makedirs(path)
        