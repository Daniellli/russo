###
###
 # @Author: xushaocong
 # @Date: 2022-10-22 01:33:28
 # @LastEditTime: 2022-10-24 00:07:15
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/omni_supervise.sh
 # email: xushaocong@stu.xmu.edu.cn
### 
 # @Author: xushaocong
 # @Date: 2022-08-21 19:15:53
 # @LastEditTime: 2022-10-19 16:10:27
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test_cls.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# * dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.



# gpu_ids="0,2,3,4,6,7,9"
# gpu_num=7
# b_size=12

gpu_ids="3,4,5,6,7"
gpu_num=5
b_size=12

# gpu_ids="0,1,2,3"
# gpu_num=4
# b_size=44

# gpu_ids="1,2,3"
# gpu_num=3;
# b_size=20;



port=29522
val_freq=1;
print_freq=100;
save_freq=$val_freq;
#* for debug 


#* for  semi supervision architecture  : step2
b_size='8,4';
resume_mode_path="pretrain/pretrain_nr3d_sr3d_sr3dplus_scanrefer_5491_39.pth"


#* for not mask 
size_consistency_weight=1e-3;
center_consistency_weight=1e-1;
token_consistency_weight=1;
query_consistency_weight=1;
text_consistency_weight=1;

rampup_length=30;
epoch=600;

# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data="sr3d nr3d scanrefer sr3d+"
test_data=nr3d
DATA_ROOT=datasets/

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    omni_train.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq $val_freq --batch_size $b_size --save_freq $save_freq --print_freq $print_freq \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --max_epoch $epoch \
    --size_consistency_weight $size_consistency_weight \
    --center_consistency_weight $center_consistency_weight \
    --token_consistency_weight $token_consistency_weight \
    --query_consistency_weight $query_consistency_weight \
    --text_consistency_weight $text_consistency_weight \
    --upload-wandb \
    --checkpoint_path $resume_mode_path \
    --rampup_length $rampup_length \
    2>&1 | tee -a logs/train_test_cls.log
    






# --lr_decay_intermediate \
# --labeled_ratio $labeled_ratio \
# --lr_decay_epochs 25 26 \


    
