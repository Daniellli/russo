
###
 # @Author: daniel
 # @Date: 2023-03-28 23:07:11
 # @LastEditTime: 2023-04-21 18:16:12
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /butd_detr/scripts/pretrain_det_scanrefer_full.sh
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
gpu_num=4;

#* batch size 
b_size=12;
port=29530

#* for  semi supervision architecture  : step1 x
topk=8;
# decay_epoch="119 136";
decay_epoch="120 140";
epoch=1000;


# resume_mode_path=/data/hdd01/xusc/exp/butd_detr/logs/bdetr/scanrefer/1681659189/ckpt_epoch_118_best.pth;
# resume_mode_path=/data/hdd01/xusc/exp/butd_detr/logs/bdetr/scanrefer/1681798762/ckpt_epoch_135_best.pth;
# resume_mode_path=/data/hdd01/xusc/exp/butd_detr/logs/bdetr/scanrefer/1681833237/ckpt_epoch_200_best.pth;





TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py \
    --data_root $DATA_ROOT --use_color --batch_size $b_size \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend \
    --query_points_obj_topk $topk \
    --max_epoch $epoch \
    --lr_decay_epochs $decay_epoch \
    --upload-wandb \
    --use-tkps --joint_det \
    2>&1 | tee -a logs/pretrain_cls.log

    

# --checkpoint_path $resume_mode_path \
#     --lr_decay_intermediate \






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


