'''
Author: xushaocong
Date: 2022-08-19 16:28:28
LastEditTime: 2022-08-19 20:02:15
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/prepare_data.py
email: xushaocong@stu.xmu.edu.cn
'''
import argparse

from src.joint_det_dataset import Joint3DDataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
args, _ = parser.parse_known_args()
Joint3DDataset(split='train', data_path=args.data_root)
Joint3DDataset(split='val', data_path=args.data_root)
