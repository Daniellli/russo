###
 # @Author: xushaocong
 # @Date: 2022-10-22 10:36:35
 # @LastEditTime: 2022-10-28 20:29:07
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/omni_full_supervse_train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

# gpu_ids="0,1,2,3"
# gpu_num=4

gpu_ids="0,1,7"
gpu_num=3



port=29530
val_freq=1;
print_freq=100;
save_freq=$val_freq;


b_size='4,2';
# resume_mode_path="pretrain/full_supervise_nr3d_sr3d_sr3dplus_scanrefer_arkitscenes_5112_cls.pth"
resume_mode_path="pretrain/pretrain_%20_4773_sr3d_cls.pth"

epoch=400;


train_data="sr3d nr3d scanrefer sr3d+"
test_data=sr3d
DATA_ROOT=datasets/
unlabel_datasets_root=datasets/arkitscenes

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    omni_full_supervse_train.py --num_decoder_layers 6 \
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
    --checkpoint_path $resume_mode_path \
    --lr_decay_intermediate \
    --lr_decay_epochs 51 61 \
    --unlabel-dataset-root $unlabel_datasets_root \
    2>&1 | tee -a logs/train_test_cls.log





# --eval \





    
