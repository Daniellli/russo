###
 # @Author: xushaocong
 # @Date: 2022-08-21 19:15:53
 # @LastEditTime: 2022-08-25 10:28:21
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test_cls.sh
 # email: xushaocong@stu.xmu.edu.cn
### 







# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

train_data="sr3d nr3d scanrefer scannet sr3d+"
test_data=scanrefer
DATA_ROOT=datasets/
gpu_ids="2,3"
gpu_num=2
# b_size=10
b_size=1
port=29522
# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq 5 --batch_size $b_size --save_freq 5 --print_freq 1 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --gpu-ids $gpu_ids \
    --max_epoch 5 \
    2>&1 | tee -a logs/train_test_cls.log
