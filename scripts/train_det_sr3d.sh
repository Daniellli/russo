
###
 # @Author: daniel
 # @Date: 2022-11-19 10:39:17
 # @LastEditTime: 2023-03-29 16:22:02
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /butd_detr/scripts/train_det_sr3d.sh
 # have a nice day
### 

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
train_data=sr3d;
test_data=sr3d
DATA_ROOT=datasets/


#* GPU id you need to run this shell 
gpu_ids="1,2,3,4,5";
gpu_num=5;





#* for not mask 
box_consistency_weight=1e-4;
box_giou_consistency_weight=1e-4;
soft_token_consistency_weight=1e-6;
object_query_consistency_weight=1;
text_token_consistency_weight=0;


rampup_length=0;#*  let it as  100  if SR3D 
ema_decay=0.999;
ema_decay_after_rampup=0.99;


port=29522

b_size='10,2';
resume_model_path=logs/bdetr/sr3d/2023-3-29-sr3d-pretrain/ckpt_epoch_28_best.pth;
labeled_ratio=0.2;
topk=8;
epoch=1000;
decay_epoch="25 26";


TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train.py --data_root $DATA_ROOT --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --batch_size $b_size --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --max_epoch $epoch \
    --box_consistency_weight $box_consistency_weight \
    --box_giou_consistency_weight $box_giou_consistency_weight \
    --soft_token_consistency_weight $soft_token_consistency_weight \
    --object_query_consistency_weight $object_query_consistency_weight \
    --text_token_consistency_weight $text_token_consistency_weight \
    --ema-decay $ema_decay \
    --ema-decay-after-rampup $ema_decay_after_rampup \
    --rampup_length $rampup_length \
    --checkpoint_path $resume_model_path \
    --lr_decay_epochs $decay_epoch \
    --lr_decay_intermediate \
    --labeled_ratio $labeled_ratio \
    --upload-wandb \
    2>&1 | tee -a logs/train_det.log

    

# --joint_det --ema-full-supervise \
#* full supervise need extra parameter : 
#* 1. --joint_det
#* 2. --ema-full-supervise
#* 3. remove: --labeled_ratio $labeled_ratio

# --upload-wandb \
# --reduce_lr \

#* 这些参数改成默认参数了
# --use_color
# --use_soft_token_loss
# --use_contrastive_align
# --self_attend
# --use-tkps
# --query_points_obj_topk $topk