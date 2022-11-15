


import os
import json
import os.path as osp

import pandas as pd 

from collections import Counter

import numpy as np

from loguru import logger





'''
description:  从SR3D作者那获取 详细的标注数据
param {*} split_name
return {*}
'''
def get_refer_it_3D(data_name,split=None):
    
    scanrefer_root="datasets/refer_it_3d"

    if split is None :
        data = pd.read_csv(osp.join(scanrefer_root,data_name+".csv"))
    else :
        data = pd.read_csv(osp.join(scanrefer_root,data_name+"_"+split+".csv"))

    # logger.info(f"len of {split_name} : {data.shape[0]}")
    # all_attrs = data.columns
    # logger.info(f" column of {split_name} : {all_attrs}")
    # logger.info(f"scene number : {len(set(data['scan_id']))}")
    # stat = Counter(data['scan_id'])
    # scane_stat = np.array([v for k ,v in stat.items()])
    # avg_sample =scane_stat.mean()
    # min_sample =scane_stat.min()
    # max_sample =scane_stat.max()

    # logger.info(f"min sample: {min_sample} \n max sample : {max_sample} \n avg sample: {avg_sample}")    
    # print(data.iloc[0,:])
    return data

def read_txt(file_name):
    with open(file_name,'r') as f:
        scan_ids = f.read().split("\n")
    logger.info(f" len of {file_name} : {len(scan_ids)}")
    return scan_ids

      

def save_txt(path,data):
    with open(path, 'w') as f:
        f.write(data)



def get_scanrefer(split=None):
    if split is not None :
        path = "datasets/scanrefer/ScanRefer_filtered_%s.json"%split
    else :
        path = "datasets/scanrefer/ScanRefer_filtered.json"

    with open (path,'r')as f :
        data =json.load(f)
    
    length = len(data)
    logger.info(f" len of {split} split : {length}")
    # all_scene = set([x['scene_id']  for x in data])
    # logger.info(f" scene number  of {split} split : {len(all_scene)}")

    # all_object_id = set([x['object_id']  for x in data])
    # logger.info(f" object number  of {split} split : {len(all_object_id)}")

    # all_anno_id = set([x['ann_id']  for x in data])
    # logger.info(f" anno number  of {split} split : {len(all_anno_id)}")

    # print(data[0])
    
    return data
