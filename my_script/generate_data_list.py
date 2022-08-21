'''
Author: xushaocong
Date: 2022-08-21 14:35:30
LastEditTime: 2022-08-21 16:54:55
LastEditors: xushaocong
Description:  下载了mini scannet for test, 生成其txt文件
FilePath: /butd_detr/my_script/generate_data_list.py
email: xushaocong@stu.xmu.edu.cn
'''
import os
import json

import os.path as osp


def generate_demo_data_txt():
    target = "data/meta_data"
    src="datasets/scans/mini_scans"
    target_train= "scandemo_train.txt"
    target_test= "scandemo_val.txt"


    all_p = sorted(os.listdir(src))
    num = len(all_p)
    train = all_p[:int(num*0.8)]
    test = all_p[int(num*0.8):]


    # print('\n'.join(test))
    with open(osp.join(target,target_train),'w') as f :
        f.write('\n'.join(train))



    with open(osp.join(target,target_test),'w') as f :
        f.write('\n'.join(test))



#* cls_results


src="datasets/scans/mini_scans"
all_p = sorted(os.listdir(src))


src_cls_results = "data/cls_results.json"
target_cls_results = "data/cls_demo_results.json"

with open(src_cls_results,'r') as f :
    src_data = json.load(f)


print(all_p)
new_data = {}
for idx,(k,v) in enumerate(src_data.items()) :
    if k in all_p:
        # print(k,idx)
        new_data[k]= v


with open(target_cls_results,'w') as f :
    json.dump(new_data,f)
    







