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
#! : 1. make log dir 
# mkdir logs

#! 2. bash to run follow line
pip install transformers
pip install plyfile
pip install h5py
pip install termcolor
pip install ipdb
sh init.sh
pip install open3d
pip install trimesh
pip install huggingface_hub

#!: 3. run follow line one by one
# huggingface-cli login
# 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

