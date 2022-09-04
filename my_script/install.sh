###
 # @Author: xushaocong
 # @Date: 2022-08-19 16:32:03
 # @LastEditTime: 2022-09-04 13:02:41
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/install.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



# conda env create -f environment.yml --name bdetr3d
# conda activate bdetr3d
#* 这里下载不成功 , 网速太慢?   -----》直接使用cerberus2
# pip install -U torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cu111
#* Compile the CUDA layers for PointNet++, which we used in the backbone network: sh init.sh

#* env install , 除了cerberus2 的环境 还需要额外安装下面环境
mkdir logs
pip install transformers
pip install plyfile
pip install h5py
pip install termcolor
pip install ipdb
sh init.sh
pip install open3d
pip install trimesh




pip install huggingface_hub
# huggingface-cli login
# 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



#* refer data 
mkdir datasets
cd datasets
ln -s  ../../data/butd_data/scanrefer scanrefer
ln -s  ../../data/butd_data/refer_it_3d refer_it_3d
mv  ../../data/butd_data/gf_detector_l6o256.pth  ./
mv  ../../data/butd_data/group_free_pred_bboxes_* ./
mv  ../../data/butd_data/pretrained ../





ln -s  ../../data/scannet/scans scans
ln -s  ../../data/scannet/scans_test scans_test


#* cache 
mv ~/exp/data/butd_data/roberta   ~/.cache/huggingface/transformers/ 






#* can not run 
# python scripts/download_scannet_files.py  > logs/install.log 

#* 
# mkdir logs
# python prepare_data.py --data_root datasets  2>&1 | tee -a logs/install.log



