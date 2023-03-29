
###
 # @Author: daniel
 # @Date: 2023-03-28 23:07:11
 # @LastEditTime: 2023-03-28 23:11:01
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /butd_detr/scripts/pretrain_det_scanrefer.sh
 # have a nice day
### 
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
train_data=scanrefer
test_data=scanrefer
DATA_ROOT=datasets/

#* GPU id you need to run this shell 
gpu_ids="0,1,2,3,4,5,6";
gpu_num=7;

#* batch size 
b_size=12;
port=29530

#* for  semi supervision architecture  : step1 x
labeled_ratio=0.2;
topk=8;

decay_epoch="65";
epoch=1000;





TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py \
    --data_root $DATA_ROOT --use_color --batch_size $b_size \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --labeled_ratio $labeled_ratio \
    --max_epoch $epoch \
    --lr_decay_epochs $decay_epoch \
    --upload-wandb \
    2>&1 | tee -a logs/pretrain_cls.log


# --lr-scheduler '
# --checkpoint_path $resume_model_path 
# --lr_decay_intermediate \

#* full supervise need extra parameter : 
#* 1. --joint_det
#* 2. remove: --labeled_ratio $labeled_ratio


#* resume model : 
#* 1. --checkpoint_path $resume_mode_path \
#* 2. choose the lr decay epoch (optional) : --lr_decay_epochs 217 227
#* 3. if item 2 is activate ,  the extra item,--lr_decay_intermediate, also is needed 
#* 4. if want to let lr decay from origin state, namely, not load the ckpt lr, add : --reduce_lr 



#* if need wandb record add parameter :
#* --upload-wandb


