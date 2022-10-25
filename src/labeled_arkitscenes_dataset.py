'''
Author: xushaocong
Date: 2022-10-22 10:41:31
<<<<<<< HEAD
LastEditTime: 2022-10-25 18:56:22
=======
LastEditTime: 2022-10-25 09:18:09
>>>>>>> f6d54c16c9417810615e7a2ea5802fff9f00e302
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/src/labeled_arkitscenes_dataset.py
email: xushaocong@stu.xmu.edu.cn
'''
'''
Author: xushaocong
Date: 2022-10-21 09:41:47
LastEditTime: 2022-10-22 02:01:51
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/src/arkitscenes_dataset.py
email: xushaocong@stu.xmu.edu.cn
'''



import os
import os.path as osp 
import random
import sys
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from six.moves import cPickle

# current = osp.dirname(osp.abspath(__file__))
# root =osp.dirname(current)
# os.chdir(root)
# sys.path.append(root)
# sys.path.append(current)
# sys.path.append(osp.join(root,'my_script'))



from data.model_util_scannet import ScannetDatasetConfig,rotate_aligned_boxes
from my_script.utils import make_dirs
import my_script.pc_utils  as pc_util
from my_script.pc_utils import write_pc_as_ply, write_ply

from loguru import logger 

from src.joint_det_dataset import rot_x,rot_y,rot_z,box2points,points2box,pickle_data,unpickle_data,get_positive_map
import torch
from transformers import RobertaTokenizerFast


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

DC = ScannetDatasetConfig()
DC18 = ScannetDatasetConfig(18)

from utils.taxonomy import ARKitDatasetConfig
ARKit_DC=ARKitDatasetConfig()

# MAX_NUM_OBJ = 64
# NUM_PROPOSAL = 256
MAX_NUM_OBJ = 132
NUM_CLASSES = 485




def is_valid_mapping_name(data_root,mapping_name):
    mapping_file = os.path.join(data_root, "data", "annotations", f"{mapping_name}.json")
    if os.stat(mapping_file).st_size < 60:
        return False
    return True




class LabeledARKitSceneDataset(Dataset):


    def __init__(self, num_points=50000,data_root='datasets/ARKitScenes',
        augment=False,debug=False,butd_cls = False ):
        

        
        #* set some parameters
        self.data_root = data_root
        self.data_path = osp.join(data_root,'dataset')
        self.num_points = num_points
        self.augment = augment
        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        self.butd_cls = butd_cls
        self.debug =debug

        #* load split datasets   
        #*       scan_names 是用来过滤数据的variables
        #todo load both of train and valid as one set 
        split_list = ['train', 'valid']
        all_scene_name = []
        self.annos= {}
        for split in split_list:

            scene_name_list,data_split_path = self.__get_scene_name(split)

            #* filter according to the scannet clss label 
            #* cache annotation 
            filename = osp.join(osp.dirname(self.data_root),f"{split}._arkitscenes.pkl")
            
            if not osp.exists(filename):
                annos= self.filter_bad_scene_and_cache(filename,scene_name_list,data_split_path)
            else:
                annos = list(unpickle_data(filename))[0]
            self.annos.update(annos)
            all_scene_name += list(annos.keys())
            
            logger.info(f"split {split } : {len(annos)} loaded")

        logger.info(f" total length  : {len(self.annos)} loaded")
        #* load  language model  
        model_path=osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),'.cache/huggingface/transformers/roberta')
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.scene_name = all_scene_name
        
        logger.info(f"LabeledARKitSceneDataset : {len(self.annos)} sample loaded, scene_name number : {len(all_scene_name)} ")
        
    def __get_scene_name(self,split):
        data_split_path = None
        if split == "train":
            data_split_path = os.path.join(self.data_path, "3dod/Training")
        else:
            data_split_path = os.path.join(self.data_path, "3dod/Validation")


        scan_names = None
        with open(os.path.join(self.data_path, f"{split}_filtered.txt"), 'r') as f:
            scan_names = sorted(f.read().strip().split('\n'))

        assert scan_names is not None 


        #* filter out unlabelled layout in valid set
        if split == "valid":
            valid_mapping = {line.split(",")[0]: line.split(",")[1] for line in open(os.path.join(self.data_root, 'data', "file.txt")).read().strip().split("\n")}
            scan_names = [scan_name for scan_name in scan_names if is_valid_mapping_name(self.data_root,valid_mapping[scan_name])]

            
        return scan_names,data_split_path
        

    '''
    description: 将一些 存在的label都不是scannet的数据给过滤掉
    param {*} self
    param {*} file_nam
    param {*} scene_names
    param {*} data_path
    return {*}
    '''
    def filter_bad_scene_and_cache(self,file_name,scene_names,data_path):
        anns = {}
        scan_box = []
        logger.info(f" before filter_bad_scene, length of scan_names :  {len(scene_names)}")
        
        for idx, scan_name in enumerate(tqdm(scene_names)):
            scan_dir = os.path.join(data_path, scan_name, f"{scan_name}_offline_prepared_data_2")
            annnotation = np.load(os.path.join(scan_dir, f"{scan_name}_label", f"{scan_name}_bbox.npy"), allow_pickle=True).item()

            anno = self.get_annos(annnotation,scan_name,data_path)#* get ann correponding to the butd detr
            if anno is not None :
                anns[scan_name]= anno
            
            #!+=================    
            # if idx == 5 :
            #     break
            #!+=================
            
        
        logger.info(f" after filter_bad_scene, length of scan_names :  {len(list(anns.keys()))}, and has cached")
        pickle_data(file_name, anns)
        return  anns

  
    '''
    description:  scannet 标签的加载function, 用于参考, 然后进一步coding
    param {*} self
    return {*}
    '''
    def get_annos(self,annotation,scan_id,data_path):
        """Load annotations of scannet."""


        # todo   过滤  不存在任何有效target的 sample 
        #*  第一步过滤
        
        class_label = annotation['types']
        class_label_in_scannet= [DC.type2class.get(label,-1) for label in class_label] #* 双向过滤1: from  arkitscene to scannet ,  -1 means not in  scannet class set, 


        #*  第二步过滤
        finnal_class_label = [] 
        for c in class_label_in_scannet:
            if c == -1:#* 第一步过滤 已经过滤掉的数据
                finnal_class_label.append(c)
                continue 
                
            if DC.class2type[c]  in  class_label and  (c  in set(DC.nyu40id2class) ): #* 第二步过滤, 如果 id 能成功在最初的arkitscene 找到对应 则 保留
                finnal_class_label.append(c)
            else :#* 否则改-1 , 表示没找到对应的数据
                finnal_class_label.append(-1)



        class_label_in_scannet = np.array(finnal_class_label)

  



        if (class_label_in_scannet!=-1).sum()>0:
            # this will get populated randomly each time
            return {
                'scan_id': scan_id,
                'target_id': [],
                'distractor_ids': [],
                'utterance': '',
                'target': [],
                'anchors': [],
                'anchor_ids': [],
                'bboxes':annotation['bboxes'],
                'bbox_class_in_arkitscenes':class_label,
                'bbox_class_ids_in_arkitscenes':np.array([ARKit_DC.cls2label[l] for l in class_label]),#* box class   id
                'bbox_class_ids_in_scannet':class_label_in_scannet,#* -1 means not exist in scannet
                'dataset': 'ARKitScenes',
                "data_path":data_path
            }
        else :
            return None


    def __len__(self):
        return len(self.scene_name )
       
    
    '''
    description: #* 去掉天花板 , 方便可视化看 
    param {*} self
    param {*} mesh_vertices
    return {*}
    '''
    def downsample_ceil(self,mesh_vertices):
        # logger.info(f"before delete ceil :{mesh_vertices.shape}")        
        delete_thres = np.percentile(mesh_vertices[..., 2], 80)
        filter_mask = (mesh_vertices[..., 2] >= delete_thres)
        new_vertices= mesh_vertices[~filter_mask].copy()
        del mesh_vertices
        mesh_vertices = new_vertices
        # logger.info(f"after  delete ceil :{mesh_vertices.shape}")
        return mesh_vertices
  
        


    '''
    description:  对其pc and box , 并且对pc 下采样
    param {*} mesh_vertices
    param {*} instance_bboxes
    return {*}
    '''
    
    def align_box_pc(self, mesh_vertices,scene_box):
         # Prepare label containers

        target_bboxes= np.copy(scene_box[:, 0:6])


        #* downsample points
        if self.debug:
            mesh_vertices=self.downsample_ceil(mesh_vertices)


        point_cloud, choices = pc_util.random_sampling(mesh_vertices,
            self.num_points, return_choices=True)    

        # ema_point_clouds, ema_choices = pc_util.random_sampling(mesh_vertices,
        #     self.num_points, return_choices=True)
        
        # TODO: OBB-Guided Scene Axis-Alignment: Rotation
        angle = np.percentile(scene_box[..., -1] % (np.pi / 2), 50)
        rot_mat = pc_util.rotz(angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
        # ema_point_clouds[:, 0:3] = np.dot(ema_point_clouds[:, 0:3], np.transpose(rot_mat))
        target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
        
        former_angle = scene_box[..., -1]
        former_angle -= angle
        former_angle %= 2 * np.pi
        reverse_mask = ((np.pi / 4 <= former_angle) & (former_angle <= np.pi / 4 * 3)) | \
            ((np.pi / 4 * 5 <= former_angle) & (former_angle <= np.pi / 4 * 7))
        padding = np.zeros((target_bboxes.shape[0]-reverse_mask.shape[0]))
        reverse_mask = np.concatenate([reverse_mask, padding], axis=0)

        dx = np.copy(target_bboxes[..., 3])
        dy = np.copy(target_bboxes[..., 4])
        target_bboxes[..., 3] = dy * reverse_mask + dx * (1-reverse_mask)
        target_bboxes[..., 4] = dx * reverse_mask + dy * (1-reverse_mask)
        
        # TODO: OBB-Guided Scene Axis-Alignment: Translation
        z_filter_L = np.percentile(point_cloud[..., 2], 15)
        z_filter_H = np.percentile(point_cloud[..., 2], 85)
        filter_mask = (point_cloud[..., 2] >= z_filter_L) & (point_cloud[..., 2] <= z_filter_H)
        x_base = np.percentile(point_cloud[filter_mask, 0], 50)
        y_base = np.percentile(point_cloud[filter_mask, 1], 50)
        z_base = np.percentile(point_cloud[..., 2], 5)
        offset = np.array([x_base, y_base, z_base])
    
        point_cloud[:, 0:3] = point_cloud[:, 0:3] - offset[None, ...]
        # ema_point_clouds[:, 0:3] = ema_point_clouds[:, 0:3] - offset[None, ...]
        target_bboxes[:, 0:3] = target_bboxes[:, 0:3] - offset[None, ...]

        #* 下移 h/2 
        target_bboxes[:,2] = target_bboxes[:,2]  - target_bboxes[:,-1]/2

        return point_cloud,target_bboxes

    

    '''
    description:  从 场景出现的 目标中选出 x 个作为utterance 
    param {*} self
    param {*} annotation
    return {*}
    '''
    def _sample_classes(self, annotation):
        """Sample classes for the scannet detection sentences."""
        
        
        sampled_classes = set(annotation['bbox_class_ids_in_scannet'][annotation['bbox_class_ids_in_scannet']!=-1]) #* 这是 当前场景对应的class id
        
        sampled_classes = list(sampled_classes & set(DC.nyu40id2class))#* 过滤一些 : 137,12 
        
        
        # sample 10 classes

        #*  默认训练才使用这个集合
        
        if len(sampled_classes) > 10:#* 如果大于10个只取是个,   对应文章生成positive and negtive sample 的方法
            sampled_classes = random.sample(sampled_classes, 10)

        ret = [DC.class2type[i] for i in sampled_classes]#* 转 成 场景的class label , 与scannet 对应的class label
       

        random.shuffle(ret)#* 随机排下序

        return ret


    def _create_scannet_utterance(self, sampled_classes):
        # if self.split == 'train' 
        
        neg_names = []#* 生成negtive sample  , 取10个? 一定取10个!!! 
        while len(neg_names) < 10:
            _ind = np.random.randint(0, len(DC.class2type))#* 随机取一个 idx, 然后判断是否是已有的positive or negtive sample 
            if DC.class2type[_ind] not in neg_names + sampled_classes:
                neg_names.append(DC.class2type[_ind])
        mixed_names = sorted(list(set(sampled_classes + neg_names)))
        random.shuffle(mixed_names)#* 打乱次序
        
        utterance = ' . '.join(mixed_names)#* 连接成一个句子
        return utterance

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        words = set(utterance.split())
        return any(rel in words for rel in rels)


    '''
    description:  
    param {*} self
    param {*} pc
    param {*} color
    param {*} rotate
    return {*}
    '''
    def _augment(self, pc, color, rotate):
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            # theta_z = 90*np.random.randint(0, 4) + (2*np.random.rand() - 1) * 5
            theta_z = 90*np.random.randint(0, 1) + (2*np.random.rand() - 1) * 5
            # Flipping along the YZ plane
            augmentations['yz_flip'] = np.random.random() > 0.5
            if augmentations['yz_flip']:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations['xz_flip'] = np.random.random() > 0.5
            if augmentations['xz_flip']:
                pc[:, 1] = -pc[:, 1]
        else:
            augmentations['yz_flip'] =False
            augmentations['xz_flip'] =False
            theta_z = (2*np.random.rand() - 1) * 5
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_y'] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations['noise'] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations['shift'] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations['shift']

        # Scale
        augmentations['scale'] = 0.98 + 0.04*np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None:
            color += self.mean_rgb
            #? do not need to record the augment parameter for color? 
            color *= 0.98 + 0.04*np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations

    def _get_pc(self, pc):
        """Return a point cloud representation of current scene."""
        

        color = np.copy(pc[:,3:])
        point_clouds = np.copy(pc[:,:3])
        del pc

        origin_color = np.copy(color)

        # d. Augmentations
        color = color - self.mean_rgb
        origin_pc = None
        if  self.debug :#* 用原来的color
            origin_pc = np.copy(np.concatenate([point_clouds,origin_color],axis=-1))
        else:
            origin_pc = np.copy(np.concatenate([point_clouds,color],axis=-1))


        augmentations = {}

        if self.augment:
            #* 不能进行rotation, 因为我们scene 没有align to origin  point 
            rotate =False
            point_clouds, color, augmentations = self._augment(point_clouds, color, rotate)
        # else :
        #     logger.info(f"no augmentation ")
            
        # e. Concatenate representations
        if color is not None:
            point_clouds = np.concatenate((point_clouds, color), 1)
        
        return point_clouds, augmentations,origin_color,origin_pc

    
    '''
    description:  获取target 对应的box  and mask
    param {*} self
    param {*} all_bboxes
    param {*} anno
    return {*}
    '''
    def _get_target_boxes(self,all_bboxes,anno):

        tids = anno['target_id']
        bboxes = np.zeros((MAX_NUM_OBJ, 6))#* MAX_NUM_OBJ==132
        bboxes[:len(tids),:] = all_bboxes[tids]
        bboxes[len(tids):, :3] = 1000
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[:len(tids)] = 1#* mast == 0 的就是 非目标, 是padding , maximun number == 132, 每个场景应该没有132个目标, 所以会有padding


        #!+========================================
        # if self.augment:  # jitter boxes
        #     bboxes[:len(tids)] *= (0.95 + 0.1*np.random.random((len(tids), 6)))
        #!+========================================


        return bboxes, box_label_mask
        
        
    '''
    description: 
    param {*} self
    param {*} anno
    return {*}
    '''
    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        
        cat_names = anno['target']
        

        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span+1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )
        positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        positive_map[:len(cat_names)] = gt_map
        return tokens_positive, positive_map


    def _get_scene_objects(self, boxes,anno):


        
        box_class_appear_in_scannet= boxes[anno['bbox_class_ids_in_scannet'] !=-1]
        box_num = box_class_appear_in_scannet.shape[0]
        
        
        all_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        all_bbox_label_mask[:box_num] = True

        all_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_bboxes[:box_num,:] = box_class_appear_in_scannet

        class_ids = np.zeros((MAX_NUM_OBJ,))
        class_ids[all_bbox_label_mask] = anno['bbox_class_ids_in_scannet'][anno['bbox_class_ids_in_scannet'] !=-1]

        #!================
        # if self.augment:
        #     all_bboxes *= (0.95 + 0.1*np.random.random((len(all_bboxes), 6)))
        #!================

        return class_ids, all_bboxes,all_bbox_label_mask

    

    '''
    description: 
    param {*} self
    param {*} boxes
    param {*} augmentations
    return {*}
    '''
    def align_box_to_pc(self,boxes,augmentations):

        #* do not transformation bbox 
        if len(augmentations.keys()) >0:  #* 不是训练的集的话 这个      augmentations 就是空
            all_det_pts = box2points(boxes).reshape(-1, 3)


            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])  
            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])

            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]

            
            all_det_pts += augmentations['shift']
            all_det_pts *= augmentations['scale']
            boxes = points2box(all_det_pts.reshape(-1, 8, 3))

        return boxes



    def __getitem__(self, idx, **kwargs):
        
        # scan_name = self.scan_names[idx]
        scan_name = list(self.annos.keys())[idx]
        anno  = self.annos[scan_name]
        # scan_name = anno['scan_id']
        
        
        
        #* load pc
        scan_dir = os.path.join(anno['data_path'], scan_name, f"{scan_name}_offline_prepared_data_2")
        mesh_vertices = np.load(os.path.join(scan_dir, f"{scan_name}_data", f"{scan_name}_pc.npy"))
        


       #* num_gt_boxes  : 感觉这个没什么用
        point_clouds,bboxes = self.align_box_pc(mesh_vertices,anno['bboxes'])

        #todo statistic data  
        #*1.  一些如果场景box  只会保留10个作为target 
        #*2.  一些 box label 没有出现在scannet 类别里也不考虑作为 target 
        sampled_classes = self._sample_classes(anno)#* 会删除一下box对应的label 
        utterance = self._create_scannet_utterance(sampled_classes)
        #* reverse all target in the scene 
        #* 对应的box的索引 ,   
        anno['target_id'] = np.where(([idd !=-1  and DC.class2type[idd] in sampled_classes  for idd in  anno['bbox_class_ids_in_scannet']   ])[:MAX_NUM_OBJ])[0].tolist()
        anno['target'] = [anno['bbox_class_in_arkitscenes'][ind]  for ind in anno['target_id'] ]
        anno['utterance'] = utterance
        
                
        point_clouds, augmentations, og_color,origin_pc=self._get_pc(point_clouds)

        #* teacher's things
        teacher_box = np.zeros((MAX_NUM_OBJ, 6))
        teacher_box[:bboxes.shape[0],:] = np.copy(bboxes)

        teacher_pc = origin_pc


        #* align box 
        bboxes = self.align_box_to_pc(bboxes,augmentations)

        
        if self.debug:
            # debug_pc = np.concatenate([origin_pc[:,:3],og_color],axis=-1)
            debug_pc = np.concatenate([point_clouds[:,:3],og_color],axis=-1) #* 用augment 之后的数据可视化检查

        

        #* 获取对应的 utterance target 对应的box, 
        #! 少了point_instance_label
        gt_bboxes, box_label_mask = self._get_target_boxes(bboxes,anno)
        tokens_positive, positive_map = self._get_token_positive_map(anno)


        #* scene gt box , already done
        class_ids,all_bboxes,all_bbox_label_mask=self._get_scene_objects(bboxes,anno)
        


        all_detected_bboxes = np.zeros(all_bboxes.shape)
        all_detected_bbox_label_mask = np.zeros(all_bbox_label_mask.shape)
        detected_class_ids = np.zeros((len(all_bboxes,)))
        if self.butd_cls:
            all_detected_bboxes = all_bboxes #? 那么  这个detected box 和 auged  pc 能对应上吗? 
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids[all_bbox_label_mask] = class_ids[class_ids !=0]



        #! 少了  all_detected_bboxes, all_detected_bbox_label_mask, detected_class_ids, detected_logits

        #* sem_cls_label 没有用! 
        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': np.zeros(MAX_NUM_OBJ).astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
        }

        

        
        ret_dict.update( {
            # Basic
            "scan_ids": scan_name,
            "point_clouds": point_clouds.astype(np.float32) if not  self.debug  else debug_pc.astype(np.float32), 
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "tokens_positive": tokens_positive.astype(np.int64),
            "positive_map": positive_map.astype(np.float32),
            "relation": ("none"),
            "target_name": anno['target'][0],
            "target_id": (anno['target_id'][0]),

            "all_bboxes": all_bboxes.astype(np.float32),
            "all_bbox_label_mask":all_bbox_label_mask.astype(np.bool8),
            "all_class_ids": class_ids.astype(np.int64),

            #! no detected results: 
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64) ,
            # "all_detected_logits": detected_logits.astype(np.float32) ,
            #*! no point_instance_label  , namely sematic results

            "distractor_ids": np.array(
                anno['distractor_ids'] + [-1] * (32 - len(anno['distractor_ids']))
            ).astype(int),
            "anchor_ids": np.array(
                anno['anchor_ids'] + [-1] * (32 - len(anno['anchor_ids']))
            ).astype(int),
            
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            #?  对应scennet calss set 的class ID
            "target_cid": (anno['bbox_class_ids_in_scannet'][anno['target_id'][0]]),
            "pc_before_aug":teacher_pc.astype(np.float32),
            "teacher_box":teacher_box.astype(np.float32),
            #! no teacher box because no  detected results, so we can only test on det setting 
            "augmentations":augmentations,
            "supervised_mask":np.array(2).astype(np.int64),#*  2 表示有标签 但是没有point_instance_label
            
        })


        return ret_dict




def save_(data,scene_name,save_root,has_color= True,flag = "debug"):
      #* for teacher or student 
      
    make_dirs(osp.join(save_root,scene_name))

    if has_color:
        write_pc_as_ply(
                    data['point_clouds'],
                    os.path.join(debug_path,scene_name, '%s_gt_%s.ply'%(scene_name,flag))
                )
    else :
        write_ply(
            data['point_clouds'],
            os.path.join(debug_path,scene_name, '%s_gt_%s.ply'%(scene_name,flag))
        )

    #* open3d  format bounding box 
    np.savetxt(os.path.join(debug_path, scene_name,'%s_box_%s.txt'%(scene_name,flag)),
            data['all_bboxes'][data['all_bbox_label_mask']],fmt='%s')




if __name__ == "__main__":
    dset = LabeledARKitSceneDataset( augment=True,data_root='datasets/arkitscenes',butd_cls = True)

    
    from tqdm import tqdm
    i=0
    debug_path = "logs/debug"
    # demo = dset.__getitem__(4)
    

    for example in tqdm(dset):
        target_num = example['box_label_mask'].astype(np.int64).sum()

        logger.info(f"target number == {target_num}")
        if len(example['target_name']) == 0:
            logger.info(example['scan_ids'])
            logger.info(f"target number :{len(example['target_name'])}")
            
        # save_(example,example['scan_ids'],debug_path)
        # print_attr_shape(example)
        # if i == 2:
        #     break
        i+=1
        
