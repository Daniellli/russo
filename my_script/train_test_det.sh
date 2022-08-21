###
 # @Author: xushaocong
 # @Date: 2022-08-21 10:26:03
 # @LastEditTime: 2022-08-21 22:24:01
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test_det.sh
 # email: xushaocong@stu.xmu.edu.cn
### 




DATA_ROOT=datasets/
gpu_ids="1,2,3,4,5"
gpu_num=5
# b_size=10
b_size=12
# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port 29520 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq 5 --batch_size $b_size --save_freq 5 --print_freq 1 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset scannet --test_dataset scannet \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --gpu-ids $gpu_ids \
    --max_epoch 50 \
    2>&1 | tee -a logs/train.log