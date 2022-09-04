###
 # @Author: xushaocong
 # @Date: 2022-09-03 18:20:38
 # @LastEditTime: 2022-09-03 18:23:37
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/tmp.sh
 # email: xushaocong@stu.xmu.edu.cn
### 




for x in $(seq 55 5 85); do 


test_model=`printf 'logs/bdetr/nr3d/1662185908/ckpt_epoch_%02d.pth' $x`;

echo $test_model;

done;