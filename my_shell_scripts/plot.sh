###
 # @Author: xushaocong
 # @Date: 2022-11-10 15:10:46
 # @LastEditTime: 2022-11-14 16:17:31
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_shell_scripts/plot.sh
 # email: xushaocong@stu.xmu.edu.cn
### 
train_data="scanrefer"
test_data=scanrefer;
DATA_ROOT=datasets/

gpu_ids="0,1,2,3,4,5,6,7"
gpu_num=8
b_size=64



port=29511
save_freq=100;
val_freq=1;
print_freq=1;


# resume_model_path=ablation/scanrefer_20%_all_consistency_loss_4106_v1.pth;
# resume_model_path=ablation/scanrefer_20%_no_soft_token_consistency_loss.pth;
# resume_model_path=ablation/scanrefer_20%_no_contrastive_consistency_loss.pth;
resume_model_path=ablation/scanrefer_20%_no_constrastive_soft_token_consistency_loss.pth

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
    --self_attend --augment_det --use-tkps \
    --query_points_obj_topk $topk \
    --max_epoch 400 \
    --checkpoint_path $resume_model_path \
    --eval \
    2>&1 | tee -a logs/train_test_cls.log


