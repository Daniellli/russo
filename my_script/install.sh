###
 # @Author: xushaocong
 # @Date: 2022-08-19 16:32:03
 # @LastEditTime: 2022-08-21 16:19:12
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
pip install transformers
pip install plyfile
pip install h5py
pip install termcolor
pip install ipdb
sh init.sh
mkdir logs



pip install huggingface_hub



#* can not run 
# python scripts/download_scannet_files.py  > logs/install.log 



#* 
python prepare_data.py --data_root datasets  2>&1 | tee -a logs/install.log



