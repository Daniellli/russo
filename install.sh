###
 # @Author: xushaocong
 # @Date: 2022-08-19 16:32:03
 # @LastEditTime: 2022-08-21 10:42:32
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/install.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



# conda env create -f environment.yml --name bdetr3d
# conda activate bdetr3d
#* 这里下载不成功 , 网速太慢?   -----》直接使用cerberus2
# pip install -U torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cu111
#* Compile the CUDA layers for PointNet++, which we used in the backbone network: sh init.sh

sh init.sh

#* can not run 
# python scripts/download_scannet_files.py  > logs/install.log 


#* env install , 除了cerberus2 的环境 还需要额外安装下面环境
pip install transformers
pip install plyfile
pip install h5py
pip install termcolor
pip install ipdb



#* 
python prepare_data.py --data_root three_d_data/mini_scans \
2>&1 | tee -a logs/install.log


# /DATA2/xusc/miniconda3/envs/cerberus2/bin/python


