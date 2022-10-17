###
 # @Author: xushaocong
 # @Date: 2022-10-14 16:25:42
 # @LastEditTime: 2022-10-16 10:01:29
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/end2end_train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=nr3d
test_data=nr3d
DATA_ROOT=datasets/

# gpu_ids="0,1,2,5"
# gpu_num=4
# b_size=12

gpu_ids="3,4,5,6,7"
gpu_num=5
b_size=12

# gpu_ids="0,1,2,3"
# gpu_num=4
# b_size=44

# gpu_ids="1,2,3"
# gpu_num=3;
# b_size=8;


port=29526
# labeled_ratio=0.2;
save_freq=5;
val_freq=5;
print_freq=100;
# print_freq=1;

# train_dist_mod.py
#* for  semi supervision architecture  : step1 
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    end2end_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq $val_freq --batch_size $b_size --save_freq $val_freq --print_freq $print_freq \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd_cls --self_attend \
    --max_epoch 400 \
    --upload-wandb \
    2>&1 | tee -a logs/train_test_cls.log
    
# --use-tkps \



# --lr_decay_epochs 25 26 \
# --dbeug \
# --consistency_weight 1e-4 \





   
          
                
                
                
                
                
                
                
                