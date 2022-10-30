###
###
 # @Author: xushaocong
 # @Date: 2022-10-23 00:39:49
 # @LastEditTime: 2022-10-30 14:49:08
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/multi_datasets_pretrain.sh
 # email: xushaocong@stu.xmu.edu.cn
### 
 # @Author: xushaocong
 # @Date: 2022-10-14 16:25:42
 # @LastEditTime: 2022-10-25 08:30:28
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/end2end_train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



# train_data="sr3d nr3d scanrefer scannet sr3d+"
# train_data="scanrefer sr3d+"
train_data="nr3d sr3d"
test_data=nr3d;
DATA_ROOT=datasets/

gpu_ids="1,2,4,5,6,8,9";
gpu_num=7;
b_size=12




port=29511
save_freq=5;
val_freq=5;
print_freq=100;

# train_dist_mod.py
resume_model_path=logs/bdetr/nr3d,sr3d+/1666886362/ckpt_epoch_75_best.pth;
#* for  semi supervision architecture  : step1 

topk=8;
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py --num_decoder_layers 6 \
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
    --max_epoch 400 \
    2>&1 | tee -a logs/train_test_cls.log









# --checkpoint_path $resume_model_path \
# --lr_decay_intermediate \
# --lr_decay_epochs 76 86 \


# --augment_det
# --query_points_obj_topk $topk 
# --use-tkps 





# --dbeug \
# --consistency_weight 1e-4 \





   
          
                
                
                
                
                
                
                
                