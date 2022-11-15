


from entity import *

class NR3DEntity(Entity):

    def __init__(self,data_name, split,ratio=None,split_root="data/meta_data"):
        super().__init__(data_name, split,ratio,split_root)
        

    '''
    description:  获取SR3D 作者划分好的 训练集 和测试集
    param {*} split
    return {*}
    '''
    def get_split_list(self):
        
        with open(osp.join(self.split_root,'%s_%s_scans.txt' % (self.data_name,self.split)),'r') as f:
            scan_ids = set(eval(f.read()))
        logger.info(f" length : {len(scan_ids)}")
        return scan_ids

       


    def get_ratio_split_list(self):

        with open(osp.join(self.split_root,'%s_%s_%.1f.txt' % (self.data_name,self.split,self.ratio)),'r') as f:
            scan_ids = f.read().split("\n")
        logger.info(f" assignment id length : {len(scan_ids)}")
        return np.array(scan_ids,dtype=np.int64).tolist()




    def get_all_ann_by_scanid_no_filter(self):
        #* 根据 scan id 来划分数据集
        annos=  get_refer_it_3D(self.data_name)
        #* 根据当前scenario 进行过滤            
        ans = []
        for idx in range(annos.shape[0]):
            ann = annos.iloc[idx]
            if ann.scan_id in self.scan_ids:
                ans.append(ann)
                     
        
        logger.info(f"annos num = {len(ans)}")
        return ans



    def get_all_ann_by_scanid(self):
        #* 根据 scan id 来划分数据集
        annos=  get_refer_it_3D(self.data_name)
        #* 根据当前scenario 进行过滤            
        ans = []
        for idx in range(annos.shape[0]):
            ann = annos.iloc[idx]
            if ann.scan_id in self.scan_ids \
                and ann.mentions_target_class and (ann.correct_guess  or self.split != 'test'):
                ans.append(ann)
        logger.info(f"annos num = {len(ans)}")             
        return ans
        




    '''
    description: 根据场景 获取对应场景的所有annotation
    param {*} self
    param {*} scene_list
    return {*}
    '''
    def get_assignment_id(self,scene_list):

        assignments = []
        
        for idx in range(len(self.annos)):
            if self.annos[idx]['scan_id'] in scene_list: 
                assignments.append (self.annos[idx]['assignmentid'])
        return assignments




    '''
    description:  加载所有的annotaion 根据 划分好的assignment id
    param {*} self
    return {*}
    '''
    def get_all_ann_by_assigmentid(self):
        annos=  get_refer_it_3D(self.data_name)
        #* 根据当前scenario 进行过滤            
        ans = []
        for idx in range(annos.shape[0]):
            ann = annos.iloc[idx]
            #* 根据assignment id 来划分数据集
            if ann.assignmentid in self.scan_ids \
                and ann.mentions_target_class and (ann.correct_guess or self.split != 'test'):
                ans.append(ann)

        logger.info(f"annos num = {len(self.annos)}")
        return ans


    def get_all_ann(self):

        annns = None
        if self.ratio is not None:
            annns=self.get_all_ann_by_assigmentid()
        else:
            annns=self.get_all_ann_by_scanid()

        self.annos = annns



    def split_by_scene_id(self,labeled_ratio):

        nr3d_ids = self.get_split_list()
        num_scans = len(nr3d_ids)
        num_labeled_scans = int(num_scans*labeled_ratio)
        choices = np.random.choice(num_scans, num_labeled_scans, replace=False)#* 从num_scans 挑选num_labeled_scans 个场景 出来 
        labeled_scan_names = list(np.array(list(nr3d_ids))[choices])
        
        save_txt(os.path.join(self.split_root,'nr3d_train_{}.txt'.format(labeled_ratio)),'\n'.join(labeled_scan_names))
        logger.info('\tSelected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names),num_scans- len(labeled_scan_names)))
    


    '''
    description:  根据比例获取不重复的assignmentid set
    param {*} self
    param {*} ratio
    return {*}
    '''
    def split_labeled_according_assignment_id(self,ratio):
        
        all_ann= self.get_all_ann_by_scanid()
        assignments = []
        for idx in range(len(all_ann)):
            assignments.append (all_ann[idx]['assignmentid'])
        

        
        length = len(assignments)
        choices = np.random.choice(length,int(length*ratio),replace= False)

        split_res = np.array(assignments)[choices]
        return split_res

    

    '''
    description:  根据assignment id 划分不同的subset 
    param {*} self
    return {*}
    '''
    def split_nr3d_according_to_assignmentid(self):
        for ratio in np.linspace(0.1,0.9,9):
            ratio = round(ratio,1)
            split_labeled_data = self.split_labeled_according_assignment_id(ratio)

            print(f"length  = {len(split_labeled_data)}")
            
            save_txt(os.path.join(self.split_root,'nr3d_train_{}.txt'.format(ratio)),'\n'.join(split_labeled_data.astype(np.str0).tolist()))
