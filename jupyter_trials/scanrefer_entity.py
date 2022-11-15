
from entity import *


class ScanReferEntity(Entity):

    def __init__(self,data_name, split,ratio=None,split_root="datasets/scanrefer",init_anno=False):
        
        super().__init__(data_name=data_name, split=split,ratio=ratio,split_root=split_root,init_anno=init_anno)


    
        
    '''
    description:  获取SR3D 作者划分好的 训练集 和测试集
    param {*} split
    return {*}
    '''
    def get_split_list(self):
        
        with open(osp.join(self.split_root,'ScanRefer_filtered_%s.txt' % (self.split) ),'r') as f:
            scan_ids =f.read().split('\n')
        logger.info(f" length : {len(scan_ids)}")
        return scan_ids

        
        


    def get_ratio_split_list(self):

        with open(osp.join(self.split_root,'ScanRefer_filtered_%s_%.1f.txt' % (self.split,self.ratio)),'r') as f:
            scan_ids = f.read().split("\n")

        # logger.info(f" length : {len(scan_ids)}")
        return scan_ids

            
        
    
    '''
    description:  根据当前scan id 获取 annotation
    param {*} self
    return {*}
    '''
    def get_all_ann_by_scanid(self):
        annns  = []
        #* 获取所有annotaiton
        scanrefer = get_scanrefer(split=self.split)


        #* 根据self.scan_ids 来过滤
        for refer in scanrefer:
            if refer['scene_id'] in self.scan_ids:
                annns.append(refer)


        logger.info(f" length : {len(annns)},ratio: {self.ratio}")
        return annns


    '''
    description:  根据过滤后的scan_ids 来获取annotation
    param {*} self
    param {*} ratio
    return {*}
    '''
    def get_all_ann(self):
        
        self.annos = self.get_all_ann_by_scanid()


            
        
    '''
    description: 
    param {*} labeled_ratio
    return {*}
    '''
    def split_by_scene_id(self,ratio):
        annos = self.get_all_ann_by_scanid()

        num_scans = len(annos)

        num_labeled_scans = int(num_scans*ratio)


        choices = np.random.choice(num_scans, num_labeled_scans, replace=False)#* 从num_scans 挑选num_labeled_scans 个场景 出来 

        labeled_scan_names = list(np.array(list(annos))[choices])
        
        with open(os.path.join(split_root,'ScanRefer_filtered_train_{}.txt'.format(ratio)), 'w') as f:
            f.write('\n'.join(labeled_scan_names))
        
        logger.info('\tSelected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names),num_scans- len(labeled_scan_names)))




    def get_scanrefer_data_by_scene(self,scene_name):

    
        res = [] 
        for line in self.annos:
            if line['scene_id'] == scene_name:
                res.append(line)
        return res
        
    def stat_object_id_for_scene(self,scene_data,object_id):
        res = [] 
        for line in scene_data:
            if line['object_id'] == object_id:
                res.append(line)
        return res

        

    def stat_ann_id_for_scene(self,scene_data,ann_id):
        res = [] 
        for line in scene_data:
            if line['ann_id'] == ann_id:
                res.append(line)
        return res
    

    def stat_ann_id_and_object_id_for_scene(self,scene_data,ann_id,object_id):
        res = [] 
        for line in scene_data:
            if line['ann_id'] == ann_id and line['object_id'] == object_id:
                res.append(line)
        return res
