'''
Author: daniel
Date: 2022-11-15 12:49:15
LastEditTime: 2022-11-15 12:54:16
LastEditors: daniel
Description: 
FilePath: /butd_detr/jupyter_trials/sr3d_entity.py
have a nice day
'''




from entity import *
class SR3DEntity(Entity):


    def __init__(self,data_name, split,ratio=None,split_root="data/meta_data"):
        super().__init__(data_name, split,ratio,split_root)

    
    def get_all_ann(self):
        annos=  get_refer_it_3D(self.data_name,self.split)

        #* 根据当前scenario 进行过滤
        ans = []
        for idx in range(annos.shape[0]):

            ann = annos.iloc[idx]

            if ann.scan_id in self.scan_ids and ann.mentions_target_class:
                ans.append(ann)
            
        self.annos = ans
        logger.info(f"annos num = {len(self.annos)}")

      