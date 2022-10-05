###
 # @Author: xushaocong
 # @Date: 2022-08-21 19:15:53
 # @LastEditTime: 2022-10-05 21:37:27
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test_cls.sh
 # email: xushaocong@stu.xmu.edu.cn
### 





# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=sr3d
test_data=sr3d
DATA_ROOT=datasets/

# gpu_ids="0,1,2,5"
# gpu_num=4
# b_size=12

# gpu_ids="0,1,2,3,4,5,6,7"
# gpu_num=8
# b_size=8

# gpu_ids="0,1,2,3"
# gpu_num=4
# b_size=44

gpu_ids="0,1,2,3"
gpu_num=4;
b_size=20;

port=29522



#* for  semi supervision architecture  : step1 
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    pretrain.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq 5 --batch_size $b_size --save_freq 5 --print_freq 100 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --max_epoch 400 --consistency_weight 1e-4 \
    --upload-wandb \
    --labeled_ratio 0.1 \
    2>&1 | tee -a logs/train_test_cls.log




#* for  semi supervision architecture  : step2
# b_size='4,16';
# resume_mode_path="pretrained/bdetr_sr3d_cls_67.1.pth"
# resume_mode_path="pretrain/pretrain_seq%50anno_60.pth"

# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
#     train.py --num_decoder_layers 6 \
#     --use_color \
#     --weight_decay 0.0005 \
#     --data_root $DATA_ROOT \
#     --val_freq 1 --batch_size $b_size --save_freq 1 --print_freq 50 \
#     --lr_backbone=1e-3 --lr=1e-4 \
#     --dataset $train_data --test_dataset $test_data \
#     --detect_intermediate --joint_det \
#     --use_soft_token_loss --use_contrastive_align \
#     --log_dir ./logs/bdetr \
#     --lr_decay_epochs 25 26 \
#     --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
#     --butd_cls --self_attend \
#     --max_epoch 400 --consistency_weight 1e-4 \
#     --upload-wandb \
#     --checkpoint_path $resume_mode_path \
#     2>&1 | tee -a logs/train_test_cls.log






#* --lr_decay_epochs 作者最开始的设置是 25 26 现在是30 35 
#*  for mean teacher  
# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
#     mean_teacher.py --num_decoder_layers 6 \
#     --use_color \
#     --weight_decay 0.0005 \
#     --data_root $DATA_ROOT \
#     --val_freq 1 --batch_size $b_size --save_freq 1 --print_freq 1 \
#     --lr_backbone=1e-3 --lr=1e-4 \
#     --dataset $train_data --test_dataset $test_data \
#     --detect_intermediate --joint_det \
#     --use_soft_token_loss --use_contrastive_align \
#     --log_dir ./logs/bdetr \
#     --lr_decay_epochs 25 26 \
#     --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
#     --butd_cls --self_attend \
#     --max_epoch 400 \
#     --checkpoint_path $resume_mode_path \
#     --consistency_weight 1e-4 \
#     --upload-wandb \
#     2>&1 | tee -a logs/train_test_cls.log


    
