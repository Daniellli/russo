

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
train_data=scanrefer
test_data=scanrefer
DATA_ROOT=datasets/


#* GPU id you need to run this shell 
gpu_ids="0,4,5,6,7";
gpu_num=5;




#* for not mask 
size_consistency_weight=1e-3;
center_consistency_weight=1e-1;
token_consistency_weight=1;
query_consistency_weight=1;
text_consistency_weight=1e-2;

rampup_length=100;#*  let it as  100  if SR3D 
ema_decay=0.99;


val_freq=1;
print_freq=100;
save_freq=$val_freq;
port=29522

epoch=800;
b_size='4,8';

resume_model_path=archive/table1_scanrefer/pretrain_50%_scanrefer_3953_192.pth;
labeled_ratio=0.5;
topk=8;
decay_epoch="375 445";

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train.py --use_color --data_root $DATA_ROOT \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth --batch_size $b_size \
    --val_freq $val_freq --save_freq $save_freq --print_freq $print_freq \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --use_soft_token_loss --use_contrastive_align \
    --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --max_epoch $epoch \
    --size_consistency_weight $size_consistency_weight \
    --center_consistency_weight $center_consistency_weight \
    --token_consistency_weight $token_consistency_weight \
    --query_consistency_weight $query_consistency_weight \
    --text_consistency_weight $text_consistency_weight \
    --ema-decay $ema_decay \
    --rampup_length $rampup_length \
    --checkpoint_path $resume_model_path \
    --lr_decay_epochs $decay_epoch \
    --lr_decay_intermediate \
    --labeled_ratio $labeled_ratio \
    2>&1 | tee -a logs/train_det.log


# --joint_det --ema-full-supervise \

#* full supervise need extra parameter : 
#* 1. --joint_det
#* 2. --ema-full-supervise
#* 3. remove: --labeled_ratio $labeled_ratio




# --upload-wandb \
# --reduce_lr \
