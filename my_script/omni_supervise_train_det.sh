###
###
 # @Author: xushaocong
 # @Date: 2022-10-26 21:42:30
 # @LastEditTime: 2022-10-30 16:37:16
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/omni_supervise_train_det.sh
 # email: xushaocong@stu.xmu.edu.cn
### 
###
 # @Author: xushaocong
 # @Date: 2022-10-22 01:33:28
 # @LastEditTime: 2022-10-26 21:33:58
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/omni_supervise_train.sh
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

#* dataset could be [sr3d, nr3d, scanrefer, scannet, sr3d+]
#!  NR3D and ScanRefer  need much more epoch for converge 
#!  To train on multiple datasets, e.g. on SR3D and NR3D simultaneously, set --TRAIN_DATASET sr3d nr3d.



# gpu_ids="0,1,2,3,4,5,6,7";
# gpu_num=8;

gpu_ids="0,1,7";
gpu_num=1;



port=29522
val_freq=1;
print_freq=1;
save_freq=$val_freq;

#* for debug


#* for  semi supervision architecture  : step2
b_size='2,1';

resume_mode_path="pretrained/scanrefer_det_52.2.pth"



#* for not mask
size_consistency_weight=1e-3;
center_consistency_weight=1e-1;
token_consistency_weight=1;
query_consistency_weight=1;
text_consistency_weight=1;

rampup_length=100;
epoch=400;

train_data=scanrefer
test_data=scanrefer
DATA_ROOT=datasets/
ema_decay=0.99;
topk=8;
unlabel_datasets_root=datasets/arkitscenes;


TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    omni_supervise_train.py --num_decoder_layers 6 \
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
    --self_attend \
    --max_epoch $epoch \
    --size_consistency_weight $size_consistency_weight \
    --center_consistency_weight $center_consistency_weight \
    --token_consistency_weight $token_consistency_weight \
    --query_consistency_weight $query_consistency_weight \
    --text_consistency_weight $text_consistency_weight \
    --rampup_length $rampup_length \
    --use-tkps \
    --ema-decay $ema_decay \
    --query_points_obj_topk $topk \
    --unlabel-dataset-root $unlabel_datasets_root \
    2>&1 | tee -a logs/train_test_cls.log

# --checkpoint_path $resume_mode_path \
# --upload-wandb \


# --lr_decay_intermediate \
# --labeled_ratio $labeled_ratio \
# --lr_decay_epochs 25 26 \

