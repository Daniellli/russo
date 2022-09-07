###
 # @Author: xushaocong
 # @Date: 2022-08-21 10:26:03
 # @LastEditTime: 2022-09-07 07:38:28
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test_det.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



train_data=scanrefer;
test_data=scanrefer;
DATA_ROOT=datasets/
gpu_ids="1,2,3,4,5,6,7"
gpu_num=7
b_size=12
port=29526
save_interval=1

resume_mode_path=logs/bdetr/scanrefer/train2/ckpt_epoch_90_best.pth
#* train
# CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
#     train_dist_mod.py --num_decoder_layers 6 \
#     --use_color \
#     --weight_decay 0.0005 \
#     --data_root $DATA_ROOT \
#     --val_freq $save_interval --batch_size $b_size --save_freq $save_interval --print_freq 10 \
#     --lr_backbone=1e-3 --lr=1e-4 \
#     --dataset $train_data --test_dataset $test_data \
#     --detect_intermediate --joint_det \
#     --use_soft_token_loss --use_contrastive_align \
#     --log_dir ./logs/bdetr \
#     --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
#     --butd --self_attend --augment_det \
#     --max_epoch 400 \
#     --upload-wandb \
#     2>&1 | tee -a logs/train.log


    


#* det : --butd --augment_det \
#* cls: --butd_cls \



#? --detect_intermediate :   是否检测utterence 中的所有目标 , 包括target and anchors
#? --joint_det :  object detecion and visual grounding 一起,   如果不给的话就是正常的 visual grounding 
#? --use_soft_token_loss  : 是否使用soft token loss, 这样每个query 都需要预测一个与 utternece tokens 长度对应的map , gt也是一个map
#? --use_contrastive_align  : 是否使用 contrastive  align loss,无监督loss, 用于拉近匹配query token pair 之间的距离
#? --butd :    
#? --butd_gt :这个参数不管是cls 还是det 都没有用到, 主要就是使用 grounding bbox , 而不是group free detector 检测的结果
#? --self_attend :    add self-attention in encoder  , 就是在BUTD-DETR 的encoder 中使用self attention 
#? --augment_det  : 对detected bbox 进行  数据增强
#? --butd_cls : cls 特指visual grounding ,
    #?  就是对检测到的bbox 进行分类 ,分类到指定的target utternece  words.  这个参数就是Assume a perfect object proposal stage ,   
    #? 就是在visual grounding 的时候, 假设这个 proposal 都是gt, 



#* resume scanfer
#* update params : lr_backbone,lr,text_encoder_lr,reduce_lr
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq $save_interval --batch_size $b_size --save_freq $save_interval --print_freq 5 \
    --lr_backbone=1e-4 --lr=1e-5 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 400 \
    --upload-wandb \
    --checkpoint_path $resume_mode_path \
    2>&1 | tee -a logs/train.log



    # --text_encoder_lr 1e-6 \
    # --reduce_lr \
#!======================================
# 2. "lr": 0.0001, 改成1e-5 
# 3. "lr_backbone": 0.001, 改成1e-4
# 4. "text_encoder_lr": 1e-05, ---> 1e-6
# 5. "lr_decay_epochs": [
#   25,
#   26
# ], -- > "lr_decay_epochs": [
#   280,
#   340
# ],
# 7. reduce_lr
#!======================================
