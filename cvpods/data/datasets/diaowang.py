from ..registry import DATASETS
from ..base_dataset import BaseDataset
import json
import os.path as osp
from tqdm import tqdm
import numpy as np
import copy
import cv2
import torch
from cvpods.structures import Instances, Boxes, BoxMode
from .paths_route import _PREDEFINED_SPLITS_DIAOWANG
import os
from pdb import set_trace

@DATASETS.register()
class DiaoWangDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=None, is_train=True):
        super(DiaoWangDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        
        self.CLASSES = ('feidiaowang','diaowang')
        self.balance_pos_neg = False
        self.cat_labels = {i:name for i,name in enumerate(self.CLASSES)}
        image_root, json_file = _PREDEFINED_SPLITS_DIAOWANG['diaowang'][dataset_name]
        self.balance_class_num = cfg.INPUT.get('balance_class_num',False)
        self.without_slides = cfg.DATASETS.get('without_slides',[])
        self.test_slides_only = cfg.DATASETS.get('test_slides_only',[])
        self.with_mask_on = cfg.DATASETS.get('WITH_MASK_ON',False)
        self.only_these_pathces = cfg.DATASETS.get('only_these_pathces',None)
        self.test_patches_only = cfg.DATASETS.get('test_patches_only',None)
        self.dataset_dicts = self._load_annotations(json_file,image_root)
        self.dataset_dicts = self._filter_annotations(filter_empty=True)
        self.one_brance_train = cfg.INPUT.get('ONE_BRANCE_TRAINING',False) if is_train else False 
        # from pdb import set_trace
        # set_trace()
        # each_patch_leng = [{osp.basename(x['file_name']).strip('.png'):len(x['annotations'])} for x in self.dataset_dicts]
        # with open('./test_patch_length.json','w') as f:
        #     json.dump({'length':each_patch_leng},f)
        self._set_group_flag()

        ## test images with no bboxes are dirty data
        print(len(self))

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, index):
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        image = cv2.imread(dataset_dict["file_name"])
        if image is None:
            image = np.zeros((512,512,3),dtype=np.uint8)

        image_shape = image.shape[:2]
        if image.shape != (512,512,3):
            image = np.zeros((512,512,3),dtype=np.uint8)
            
        instances = Instances(image_shape)

        if len(dataset_dict['annotations']) == 0:
            annotations = []
            image, _ = self._apply_transforms(image, None)

        else:
            annotations = []
            for ann in dataset_dict['annotations']:

                annotations.append({'bbox':np.array(ann['bbox'],dtype=np.float32),
                                'category_id':np.array(ann['category_id'],dtype=np.int64),'bbox_mode':BoxMode.XYXY_ABS})
        
            image, annotations = self._apply_transforms(image, annotations)
            boxes = [
                BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
                for obj in annotations
            ]
            boxes = Boxes(boxes)
            boxes.clip(image_shape)
            instances.gt_boxes = boxes
            classes = [obj["category_id"] - 0 for obj in annotations]
            classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_classes = classes

        dataset_dict["instances"] = instances
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["length"] = torch.tensor(len(instances))
        return dataset_dict

    def _load_annotations(self,
                          json_file,
                          image_root):
        print('loading annotations...')
        with open(json_file,'r') as f:
            self.data = json.load(f)

        dataset_dicts = []

        if self.test_patches_only is None:
            only_these_pathces = []
        elif os.path.isdir(self.test_patches_only):
            def get_all_filenames(folder_path):
                filenames = []
                for root, dirs, files in os.walk(folder_path):
                    if 'HE' in root:
                        for file in files:
                            filenames.append(file)
                return filenames
            
            only_these_pathces = get_all_filenames(self.test_patches_only)
        elif os.path.isfile(self.test_patches_only):
            assert self.test_patches_only.endswith('.json')
            with open(self.test_patches_only) as f:
                only_these_pathces = [x+'.png' for x in json.load(f)['patch_names']]


        all_slides_name = []
        no_considering_images=[]
        bar = tqdm(self.data.items())

        for slide_name, v_s in bar:
            if slide_name in self.without_slides:
                continue

            if len(self.test_slides_only)>0 and not self.is_train and slide_name not in self.test_slides_only:
                continue

            all_slides_name.append(slide_name)
            for patch_name,v_p in v_s.items():
                if len(only_these_pathces) >0:
                    if patch_name+'.png' not in only_these_pathces:
                        continue

                dataset_dict = dict()
                img_path = osp.join(image_root, slide_name,'he_patches','Hes_512',patch_name+'.png')

                if not osp.exists(img_path):
                    img_path = osp.join('/data5/wulianjun/data', slide_name,'he_patches','Hes_512_128',patch_name+'.png')

                if osp.exists(img_path):
                    if self.balance_class_num and self.is_train:
                        if 1 not in v_p['labels']:
                            continue

                    if not self.is_train and 'labels' not in v_p:
                        continue
                        # v_p['labels'] = np.zeros(len(v_p['bboxes']),dtype=np.int64)

                    dataset_dict["file_name"] = img_path
                    dataset_dict["annotations"] = [{'bbox':bbox,'category_id':label,'bbox_mode':BoxMode.XYXY_ABS} for bbox,label in zip(v_p['bboxes'],v_p['labels'])]
                    dataset_dict['group_id'] = 1 if 1 in v_p['labels'] else 0
                    dataset_dict['species'] = 'Mouse' if 'LLC' in patch_name or 'Liu' in patch_name else 'Rabbit'
                    
                    if len(dataset_dict["annotations"])<=20:
                        continue
                    if self.with_mask_on:
                        # if not self.is_train:
                        if not osp.exists(osp.join(image_root,slide_name,'he_patches','masks',patch_name+'.png')):
                            continue
                        mask = cv2.imread(osp.join(image_root,slide_name,'he_patches','masks',patch_name+'.png'))
                        if 1 not in v_p['labels']:
                            maks_labels = np.zeros_like(mask[...,0],dtype=np.int64) - 1 
                            maks_labels[mask[...,0]==255] = 0
                        elif 0 not in v_p['labels']:
                            maks_labels = np.zeros_like(mask[...,0],dtype=np.int64) - 1 
                            maks_labels[mask[...,0]==255] = 1
                        else:
                            maks_labels_0 = np.zeros_like(mask[...,0],dtype=np.int64) - 1 
                            maks_labels_1 = np.zeros_like(mask[...,0],dtype=np.int64) - 1 
                            for bbox,label in zip(v_p['bboxes'],v_p['labels']):
                                if label == 0:
                                    maks_labels_0[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 0
                                else:
                                    maks_labels_1[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
                            maks_labels = np.zeros_like(mask[...,0],dtype=np.int64) - 1 
                            maks_labels[mask[...,0]==255] = -(maks_labels_0[mask[...,0]==255] * maks_labels_1[mask[...,0]==255])
               
                        dataset_dict['maks_labels'] = maks_labels
                
                    dataset_dicts.append(dataset_dict)
                else:
                    no_considering_images.append(patch_name)

        # for debug
        # dataset_dicts = dataset_dicts[:20]
        # from pdb import set_trace
        # set_trace()
        del self.data
        print(len(dataset_dicts))
        return dataset_dicts
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        if "width" in self.dataset_dicts[0] and "height" in self.dataset_dicts[0]:
            for i in range(len(self)):
                dataset_dict = self.dataset_dicts[i]
                if dataset_dict['width'] / dataset_dict['height'] > 1:
                    self.aspect_ratios[i] = 1
    

    def get_cat2imgs(self,seed=0):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        raise NotImplementedError
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(self.CLASSES)+1)}
        if len(self.CLASSES) ==2:
            for i in range(len(self)):
                cat_ids = self.dataset_dicts[i]['group_id']
                for cat in cat_ids:
                    cat2imgs[cat].append(i)
            if self.balance_pos_neg:
                cat2imgs[0] = cat2imgs[0] * 2
                positive_num = len(cat2imgs[0])
                hard_negtive_num = len(cat2imgs[1])
                assert positive_num >= hard_negtive_num
                negtive_num = positive_num - hard_negtive_num
                negtive_num = min(max(negtive_num,0),len(cat2imgs[2]))
                negtive = np.array(cat2imgs[2])
                np.random.seed(seed)
                np.random.shuffle(negtive)
                cat2imgs[2] = negtive[:negtive_num].tolist()
        elif len(self.CLASSES) ==1:
            for i in range(len(self)):
                cat_ids = self.dataset_dicts[i]['group_id']
                if 0 in cat_ids:
                    cat2imgs[0].append(i)
                else:
                    cat2imgs[1].append(i)
            if self.balance_pos_neg:
                cat2imgs[0] = cat2imgs[0] * 2
                negtive_num = len(cat2imgs[0])
                negtive = np.array(cat2imgs[1])
                np.random.seed(seed)
                np.random.shuffle(negtive)
                cat2imgs[1] = negtive[:negtive_num].tolist()
        return cat2imgs