###
 # @Author: xushaocong
 # @Date: 2022-08-21 22:20:53
 # @LastEditTime: 2022-08-21 23:16:44
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 








DATA_ROOT=datasets/
gpu_ids="1"
gpu_num=1
# b_size=10
b_size=2
# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port 29521 \
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
    --eval \
    --checkpoint_path pretrained/sr3d_butd_det_52.1_27.pth \
    2>&1 | tee -a logs/test.log



# --checkpoint_path /home/DISCOVER_summer2022/xusc/exp/butd_detr/logs/bdetr/scannet/1661089930/ckpt_epoch_last.pth \
