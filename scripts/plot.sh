
###
 # @Author: daniel
 # @Date: 2023-04-09 23:42:32
 # @LastEditTime: 2023-04-09 23:46:44
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /butd_detr/scripts/plot.sh
 # have a nice day
### 



#* dataset you want to train ,  could be nr3d or sr3d ,for cls 
train_data=scanrefer
test_data=scanrefer
DATA_ROOT=datasets/

#* GPU id you need to run this shell 
gpu_ids="7";
gpu_num=1;
b_size=1;
#* batch size 
port=29524



TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node $gpu_num --master_port $port \
    train_dist_mod.py \
    --data_root $DATA_ROOT --batch_size $b_size \
    --pp_checkpoint $DATA_ROOT/gf_detector_l6o256.pth \
    --dataset $train_data --test_dataset $test_data \
    --detect_intermediate --checkpoint_path "archive/table1_scanrefer/scanrefer100_all_consistency_5221_138.pth" \
    --eval --use-tkps \
    2>&1 | tee -a logs/eval.log