
###
 # @Author: daniel
 # @Date: 2022-11-14 22:15:44
<<<<<<< HEAD
 # @LastEditTime: 2022-11-21 21:32:51
=======
 # @LastEditTime: 2022-11-19 11:30:07
>>>>>>> ce5c24fe3f595e8b29416025aaf57774ea3d2673
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /butd_detr/my_shell_scripts/pretrain_cls.sh
 # have a nice day
### 

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
train_data=nr3d
test_data=nr3d
DATA_ROOT=datasets/

#* GPU id you need to run this shell 
<<<<<<< HEAD
gpu_ids="0,1,2,3,4,5,6,7";
gpu_num=8;
=======
gpu_ids="1,2,3,5,7";
gpu_num=5;
>>>>>>> ce5c24fe3f595e8b29416025aaf57774ea3d2673

#* batch size 
b_size=12;



port=29530
val_freq=5;
print_freq=10;
save_freq=$val_freq;


#* for  semi supervision architecture  : step1 x
<<<<<<< HEAD
labeled_ratio=0.5;
=======
labeled_ratio=0.2;
>>>>>>> ce5c24fe3f595e8b29416025aaf57774ea3d2673
topk=8;


TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py \
    --data_root $DATA_ROOT --use_color --batch_size $b_size \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --val_freq $val_freq --save_freq $val_freq --print_freq $print_freq \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --butd_cls --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --labeled_ratio $labeled_ratio \
    --upload-wandb \
    2>&1 | tee -a logs/pretrain_cls.log





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


