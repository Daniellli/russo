'''
Author: daniel
Date: 2022-11-15 12:48:45
LastEditTime: 2022-11-15 12:53:18
LastEditors: daniel
Description: 
FilePath: /butd_detr/jupyter_trials/entity.py
have a nice day
'''


import os
import json
import os.path as osp

import pandas as pd 

from collections import Counter

import numpy as np

from loguru import logger

from data_util import *




class Entity:

    def __init__(self,data_name, split,ratio=None,init_anno=False,split_root="data/meta_data"):

        
        self.split_root =  split_root
        self.data_name=data_name
        self.split=split
        self.ratio=ratio

        scan_ids=None
        if ratio is None:
            scan_ids=self.get_split_list()
        else :
            scan_ids=self.get_ratio_split_list()

        self.scan_ids = scan_ids


        if init_anno:
            self.get_all_ann()


        
    '''
    description:  获取SR3D 作者划分好的 训练集 和测试集
    param {*} split
    return {*}
    '''
    def get_split_list(self):
        with open(osp.join(self.split_root,'%s_%s_scans.txt' % (self.data_name,self.split) ),'r') as f:
            scan_ids = set(eval(f.read()))
        logger.info(f" length : {len(scan_ids)}")
        return scan_ids

        
        


    def get_ratio_split_list(self):
        with open(osp.join(self.split_root,'%s_%s_%.1f.txt' % (self.data_name,self.split,self.ratio)),'r') as f:
            scan_ids = f.read().split("\n")
        logger.info(f" length : {len(scan_ids)}")
        return scan_ids



    def get_all_ann(self):

      raise NotImplemented
    

    def get_scene_data(self,scene_name):
        
        return_data = [] 
        for  idx in range(self.annos.shape[0]):

            if self.annos.iloc[idx]['scan_id'] == scene_name:
                return_data.append(self.annos.iloc[idx])

        return return_data

        
        
