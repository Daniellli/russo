###
 # @Author: xushaocong
 # @Date: 2022-08-21 10:26:03
 # @LastEditTime: 2022-09-02 22:45:30
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test_det.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=nr3d;
test_data=nr3d;
DATA_ROOT=datasets/
# gpu_ids="1,2"
gpu_ids="0,3,5"
gpu_num=3
b_size=8
port=29526

#* train
# TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq 5 --batch_size $b_size --save_freq 5 --print_freq 1 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 50 \
    --upload-wandb \
    2>&1 | tee -a logs/train.log


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







