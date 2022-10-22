###
 # @Author: xushaocong
 # @Date: 2022-10-22 10:36:35
 # @LastEditTime: 2022-10-23 00:36:07
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/full_supervise.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

# gpu_ids="0,2,3,4,6,7,9"
# gpu_num=7
# b_size=12

gpu_ids="0,1,2,3"
gpu_num=4
b_size=16

# gpu_ids="0,1,2,3"
# gpu_num=4
# b_size=44

# gpu_ids="1,2,3"
# gpu_num=3;
# b_size=20;



port=29530
val_freq=5;
print_freq=10;
save_freq=$val_freq;
#* for debug 


#* for  semi supervision architecture  : step2
b_size='14,2';
# resume_mode_path="pretrain/butd_no_tkps_5284_sr3d_nr3d_scanrefer_sr3dplus_74.pth"
epoch=400;
# train_data="sr3d nr3d scanrefer scannet sr3d+"
#todo 将scanrefer 和 joint datasets 整合起来

train_data="sr3d nr3d scanrefer sr3d+"
test_data=scanrefer
DATA_ROOT=datasets/
# unlabeled_datastes='arkitscenes';


TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    omni_supervse_train.py --num_decoder_layers 6 \
    --use_color --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq $val_freq --batch_size $b_size --save_freq $save_freq --print_freq $print_freq \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --max_epoch $epoch --upload-wandb \
    2>&1 | tee -a logs/train_test_cls.log

# --checkpoint_path $resume_mode_path \
# --lr_decay_intermediate \
# --labeled_ratio $labeled_ratio \
# --lr_decay_epochs 25 26 \


    
