
###
 # @Author: daniel
 # @Date: 2022-11-21 16:48:48
 # @LastEditTime: 2022-12-07 16:06:08
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /butd_detr/my_shell_scripts/train_cls.sh
 # have a nice day
### 

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
train_data=sr3d
test_data=sr3d
DATA_ROOT=datasets/


#* GPU id you need to run this shell 
gpu_ids="0,1,3,4,5,6";
gpu_num=6;




#* for not mask 
size_consistency_weight=1e-4;
center_consistency_weight=1e-4;
token_consistency_weight=1e-2;
query_consistency_weight=1;
text_consistency_weight=1;

rampup_length=0;#*  let it as  100  if SR3D 
ema_decay=0.99;
ema_decay_after_rampup=0.99;


val_freq=1;
print_freq=50;
save_freq=$val_freq;
port=29522

epoch=1000;
b_size='6,1';


# resume_model_path=archive/nr3d/pretrain_30%_nr3d_3455_180.pth;
# resume_model_path=archive/table2_sr3d/pretrain_20%_sr3d_4456_55.pth;
resume_model_path=logs/bdetr/sr3d/1670258911/sr20_all_consistency_5345_80_1.pth;



labeled_ratio=0.2;
topk=8;
decay_epoch="1000 1001";

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train.py --data_root $DATA_ROOT \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth --batch_size $b_size \
    --val_freq $val_freq --save_freq $save_freq --print_freq $print_freq \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --butd_cls --max_epoch $epoch \
    --size_consistency_weight $size_consistency_weight \
    --center_consistency_weight $center_consistency_weight \
    --token_consistency_weight $token_consistency_weight \
    --query_consistency_weight $query_consistency_weight \
    --text_consistency_weight $text_consistency_weight \
    --ema-decay $ema_decay \
    --ema-decay-after-rampup $ema_decay_after_rampup \
    --rampup_length $rampup_length \
    --checkpoint_path $resume_model_path \
    --lr_decay_epochs $decay_epoch \
    --query_points_obj_topk $topk \
    --lr_decay_intermediate \
    --labeled_ratio $labeled_ratio \
    --upload-wandb \
    2>&1 | tee -a logs/train_cls.log


# --joint_det --ema-full-supervise \
#* full supervise need extra parameter : 
#* 1. --joint_det
#* 2. --ema-full-supervise
#* 3. remove: --labeled_ratio $labeled_ratio




# --upload-wandb \
# --reduce_lr \
