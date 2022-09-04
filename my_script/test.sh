###
 # @Author: xushaocong
 # @Date: 2022-08-21 22:20:53
 # @LastEditTime: 2022-09-03 20:58:07
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# train_data="sr3d nr3d scanrefer scannet sr3d+"
train_data=scanrefer
test_data=scanrefer
DATA_ROOT=datasets/
gpu_ids="1,2"
gpu_num=2
b_size=4
port=29527
#* test 
# test_model=pretrained/bdetr_nr3d_43.3.pth;
# test_model=pretrained/bdetr_nr3d_cls_55_4.pth;
test_model=pretrained/scanrefer_det_52.2.pth;
# test_model=pretrained/bdetr_sr3d_cls_67.1.pth;
# test_model=logs/bdetr/sr3d/train1/ckpt_epoch_95.pth;

#* test single model 
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root $DATA_ROOT \
    --val_freq  --batch_size $b_size --save_freq 5 --print_freq 1 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --eval \
    --checkpoint_path $test_model \
    2>&1 | tee -a logs/test.log


#* test multiple model 
# for x in $(seq 85 5 85); do 
# test_model=`printf 'logs/bdetr/nr3d/1662185908/ckpt_epoch_%02d.pth' $x`;
# echo "test model :$test_model";

# CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
#     train_dist_mod.py --num_decoder_layers 6 \
#     --use_color \
#     --weight_decay 0.0005 \
#     --data_root $DATA_ROOT \
#     --val_freq  --batch_size $b_size --save_freq 5 --print_freq 1 \
#     --lr_backbone=1e-3 --lr=1e-4 \
#     --dataset $train_data --test_dataset $test_data \
#     --detect_intermediate --joint_det \
#     --use_soft_token_loss --use_contrastive_align \
#     --log_dir ./logs/bdetr \
#     --lr_decay_epochs 25 26 \
#     --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
#     --butd --self_attend --augment_det \
#     --eval \
#     --checkpoint_path $test_model \
#     2>&1 | tee -a logs/test.log
# done;

#* det : --butd --augment_det \
#* cls: --butd_cls \

