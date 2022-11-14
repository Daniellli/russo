

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.

# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=nr3d
test_data=nr3d
DATA_ROOT=datasets/


gpu_ids="0,4,5,6,7";
gpu_num=5;
b_size=64



port=29530
val_freq=1;
print_freq=1;
save_freq=100;
resume_mode_path=archive/table3_nr3d/nr100_all_consistency_5390_306.pth;
topk=8;


TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --use_color --data_root $DATA_ROOT \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --val_freq $val_freq --save_freq $val_freq --print_freq $print_freq \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --batch_size $b_size \
    --use_soft_token_loss --use_contrastive_align \
    --butd_cls --self_attend --use-tkps \
    --query_points_obj_topk $topk \
    --checkpoint_path $resume_mode_path --eval \
    2>&1 | tee -a logs/eval_cls.log


