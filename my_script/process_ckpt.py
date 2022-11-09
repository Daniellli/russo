import os
import os.path as osp
from glob import glob



def delete_ckpt_except_last_one(path):
    for o in os.listdir(path ):
        all_ckpt = glob(osp.join(path,o)+"/*.pth")
        num = len(all_ckpt)
        if num >1:
            all_ckpt = sorted(all_ckpt,key = lambda x : int(x.split('/')[-1].split('_')[-2]))
            
            for ckpt in all_ckpt[:-1]:
                os.remove(ckpt)
                print(ckpt,"has deleted")
            print("=======================================")

path = "/home/DISCOVER_summer2022/xusc/exp/butd_detr/logs/bdetr/scanrefer"

delete_ckpt_except_last_one(path)


