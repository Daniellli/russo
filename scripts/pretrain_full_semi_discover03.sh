
###
 # @Author: daniel
 # @Date: 2023-03-28 23:07:11
 # @LastEditTime: 2023-04-29 14:42:43
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /russo/scripts/pretrain_full_semi_discover03.sh
 # have a nice day
### 
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.
#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
data=sr3d;
gpu_ids="0,1,2,3,4,5,6,7";
gpu_num=8;
b_size=8;
#* for  semi supervision architecture  : step1 x
labeled_ratio=0.2;
topk=8;
# decay_epoch="120 140"; #*scanrefer
# decay_epoch="81 89"; #*sr3d
decay_epoch="72"; #*sr3d

#!=====================================================================================!#
#!======================  labeled_ratio data supervised pretrain ======================!#
#!=====================================================================================!#


# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids \
#     python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $RANDOM \
#     train_dist_mod.py  --batch_size $b_size --dataset $data --test_dataset $data \
#     --detect_intermediate --use_soft_token_loss --use_contrastive_align --use_color \
#     --self_attend --query_points_obj_topk $topk \
#     --lr_decay_epochs $decay_epoch --use-tkps \
#     --labeled_ratio $labeled_ratio --wandb \
#     2>&1 | tee -a logs/pretrain2.log




# --checkpoint_path $resume_model_path 
# --lr_decay_intermediate \

#!=====================================================================================!#
#!============================== full supervised pretrain =============================!#
#!=====================================================================================!#

resume_model_path=logs/bdetr/sr3d/2023-04-27-16:17/ckpt_epoch_71_best.pth;
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $RANDOM \
    train_dist_mod.py --use_color --batch_size $b_size \
    --dataset $data --test_dataset $data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend --query_points_obj_topk $topk \
    --lr_decay_epochs $decay_epoch \
    --use-tkps --joint_det --wandb \
    --lr_decay_intermediate --checkpoint_path $resume_model_path \
    2>&1 | tee -a logs/pretrain_full2.log








#!=====================================================================================!#
#!============================== semi-supervised training =============================!#
#!=====================================================================================!#

#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
# box_consistency_weight=1e-4;
# box_giou_consistency_weight=1e-4;
# soft_token_consistency_weight=1e-6;
# object_query_consistency_weight=1;
# text_token_consistency_weight=0;
# rampup_length=0;#*  let it as  100  if SR3D 
# ema_decay=0.999;
# ema_decay_after_rampup=0.99;
# b_size='8,4';
# resume_model_path=logs/bdetr/scanrefer/1681911129/ckpt_epoch_70_best.pth;
# topk=8;

# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids \
#     python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $RANDOM \
#     train.py --semi_batch_size $b_size --dataset $data --test_dataset $data \
#     --detect_intermediate --use_contrastive_align --use_soft_token_loss --self_attend --use_color \
#     --box_consistency_weight $box_consistency_weight --box_giou_consistency_weight $box_giou_consistency_weight \
#     --soft_token_consistency_weight $soft_token_consistency_weight --ema-decay $ema_decay \
#     --object_query_consistency_weight $object_query_consistency_weight --text_token_consistency_weight $text_token_consistency_weight \
#     --ema-decay-after-rampup $ema_decay_after_rampup --rampup_length $rampup_length \
#     --lr_decay_intermediate --checkpoint_path $resume_model_path \
#     --labeled_ratio $labeled_ratio --lr_decay_epochs $decay_epoch --use-tkps \
#     2>&1 | tee -a logs/train2.log


# --wandb




#* full supervise need extra parameter : 
#* 1. --joint_det
#* 2. remove: --labeled_ratio $labeled_ratio


#* resume model : 
#* 1. --checkpoint_path $resume_mode_path \
#* 2. choose the lr decay epoch (optional) : --lr_decay_epochs 217 227
#* 3. if item 2 is activate ,  the extra item,--lr_decay_intermediate, also is needed 
#* 4. if want to let lr decay from origin state, namely, not load the ckpt lr, add : --reduce_lr 
#* if need wandb record add parameter :
#* -wandb
