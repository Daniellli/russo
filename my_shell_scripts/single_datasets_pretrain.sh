
###
 # @Author: xushaocong
 # @Date: 2022-10-24 00:27:51
 # @LastEditTime: 2022-11-09 23:29:58
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/single_datasets_pretrain_1.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=nr3d
test_data=nr3d
DATA_ROOT=datasets/


gpu_ids="0,1,2,3,4,5,6,7";
gpu_num=8;
b_size=12



port=29530
save_freq=5;
val_freq=5;
print_freq=10;
save_freq=$val_freq;

# resume_mode_path=logs/bdetr/sr3d/1667060778/ckpt_epoch_60_best.pth;
# resume_mode_path=logs/bdetr/nr3d/1667666279/ckpt_epoch_216_best.pth;


# resume_mode_path=logs/bdetr/sr3d/1667007312/ckpt_epoch_20_best.pth;


#* for  semi supervision architecture  : step1 x
labeled_ratio=0.7;

topk=8;
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq $val_freq --batch_size $b_size --save_freq $val_freq --print_freq $print_freq \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --max_epoch 400 \
    --upload-wandb \
    --labeled_ratio $labeled_ratio \
    2>&1 | tee -a logs/train_test_cls.log

    # --reduce_lr \
    # --checkpoint_path $resume_mode_path \
# --lr_decay_epochs 217 227 \
#     --lr_decay_intermediate \



# --checkpoint_path $resume_mode_path \
#     --lr_decay_epochs 61 66 \
#     --lr_decay_intermediate \

# --debug \
#  --joint_det
# --use-tkps \

