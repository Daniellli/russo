###
 # @Author: xushaocong
 # @Date: 2022-10-23 11:57:16
 # @LastEditTime: 2022-11-10 11:08:33
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train.sh
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

gpu_ids="1,2,4,5,6,7";
gpu_num=6;




#* for not mask 
size_consistency_weight=1e-5;
center_consistency_weight=5e-3;
token_consistency_weight=1e-1;
query_consistency_weight=1e-1;
text_consistency_weight=1e-2;




val_freq=1;
print_freq=10;
save_freq=$val_freq;

rampup_length=30;
epoch=400;
port=29522


#* for  semi supervision architecture  : step2
b_size='4,8';
# resume_model_path=pretrain/pretrain_nr3d_30%_3481_220.pth;
resume_model_path=logs/bdetr/nr3d/1668010407/student_ckpt_epoch_276_best.pth;

labeled_ratio=0.3;
ema_decay=0.99;
topk=8;

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq $val_freq --batch_size $b_size --save_freq $save_freq --print_freq $print_freq \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --max_epoch $epoch \
    --size_consistency_weight $size_consistency_weight \
    --center_consistency_weight $center_consistency_weight \
    --token_consistency_weight $token_consistency_weight \
    --query_consistency_weight $query_consistency_weight \
    --text_consistency_weight $text_consistency_weight \
    --rampup_length $rampup_length \
    --ema-decay $ema_decay \
    --checkpoint_path $resume_model_path \
    --lr_decay_intermediate \
    --lr_decay_epochs 277 350 \
    --labeled_ratio $labeled_ratio \
    --upload-wandb \
    2>&1 | tee -a logs/train_test_cls.log






