
###
 # @Author: xushaocong
 # @Date: 2022-11-09 09:10:25
 # @LastEditTime: 2022-11-14 17:05:56
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_shell_scripts/eval_det.sh
 # email: xushaocong@stu.xmu.edu.cn
### 
###
 # @Author: xushaocong
 # @Date: 2022-11-07 19:54:41
 # @LastEditTime: 2022-11-07 20:00:27
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/eval_nr3d.sh
 # email: xushaocong@stu.xmu.edu.cn
### 
###
 # @Author: xushaocong
 # @Date: 2022-10-24 00:27:51
 # @LastEditTime: 2022-10-30 10:11:33
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/single_datasets_pretrain.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=scanrefer
test_data=scanrefer
DATA_ROOT=datasets/

gpu_ids="0,4,5,6,7";
gpu_num=5;
b_size=64

port=29540

val_freq=1;
print_freq=1;
save_freq=100;
resume_mode_path=archive/table1_scanrefer/pretrain_100%_scanrefer_5191_112.pth;
topk=8;
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py \
    --use_color --data_root $DATA_ROOT \
    --val_freq $val_freq --batch_size $b_size --save_freq $val_freq --print_freq $print_freq \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --checkpoint_path $resume_mode_path \
    --eval \
    2>&1 | tee -a logs/eval_det.log




