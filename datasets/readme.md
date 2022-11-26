<!--
 * @Author: xushaocong
 * @Date: 2022-09-02 15:56:35
 * @LastEditTime: 2022-09-02 16:04:45
 * @LastEditors: xushaocong
 * @Description: 
 * @FilePath: /butd_detr/datasets/readme.md
 * email: xushaocong@stu.xmu.edu.cn
-->


## 一、scanrefer

 
分为 
训练集(ScanRefer_filtered_train.json,ScanRefer_filtered_train.txt),
验证集(ScanRefer_filtered_val.json,ScanRefer_filtered_val.txt),
还有一个没有划分的完整集合(ScanRefer_filtered.json)




### json file
就是主要的标签文件
主要用下面这些信息, 
"scene_id": 文件对应的scenario name
"object_id":   目标的id
"object_name":  object 对应的name, 或者class , 比如chair 
"ann_id": 标签的id
"description":  utterence, 
"token":  划分后的utterence s




### txt file
就是与标签 文件对应的scenario name list 



## 二、refer_it_3d













