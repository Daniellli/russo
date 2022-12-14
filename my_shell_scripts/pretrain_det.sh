
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

#* batch size 
b_size=12;



port=29530
val_freq=5;
print_freq=10;
save_freq=$val_freq;


#* for  semi supervision architecture  : step1 x
labeled_ratio=0.7;
topk=8;


TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py \
    --data_root $DATA_ROOT --use_color --batch_size $b_size \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --val_freq $val_freq --save_freq $val_freq --print_freq $print_freq \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --labeled_ratio $labeled_ratio \
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


