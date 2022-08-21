###
 # @Author: xushaocong
 # @Date: 2022-08-21 10:26:03
 # @LastEditTime: 2022-08-21 10:32:04
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /butd_detr/my_script/train_test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


sh scripts/train_test_det.sh 2>&1 | tee -a logs/train.log


# sh scripts/train_test_cls.sh 