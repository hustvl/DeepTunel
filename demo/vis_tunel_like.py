import os
import cv2
import json
import numpy as np
from pdb import set_trace
# from cellpose import io, models
from tqdm import tqdm
import shutil

vis_patch = ['6-3_5_50.png','6-3_5_51.png','6-3_5_52.png',
             '6-3_6_50.png','6-3_6_51.png','6-3_6_52.png',
             '6-3_7_50.png','6-3_7_51.png','6-3_7_52.png',
             '6-3_8_50.png','6-3_8_51.png','6-3_8_52.png',]

os.makedirs('./vis',exist_ok=True)
# model = models.Cellpose(gpu=True, model_type='nuclei', net_avg=True)

def get_slide_name(image_name:str):
    ret = ''
    for x in image_name.split('_')[:-2]:
        ret += x + '_'
    return ret[:-1]


def show(img,name):
    cv2.imwrite(f'./vis/{name}',img)

# test datasets predictions
with open('results_demo_like_inputs.json') as f:
    predictions = json.load(f)

for x in tqdm(vis_patch):
    slide_name = get_slide_name(x)
    if not os.path.exists(f'HE_patches/{x}'):
        continue
    he = cv2.imread(f'HE_patches/{x}')

    # masks, flows, styles, diams = model.eval(he, channels=[3,0], diameter=20, invert=True)
    masks = np.load('HE_patches/{}'.format(x.replace('.png','.npy')))
    he_ = he.copy()

    anns = predictions[slide_name][x.strip('.png')]

    aa = np.zeros((512,512,3),dtype=np.uint8)
    aa[:,:,0] = 255
    he_[masks>0] = aa[masks>0]

    for bbox, label in zip(anns['bboxes'],anns['labels']):
        aa = np.zeros((512,512,3),dtype=np.uint8)
        if label == 1:
            aa[:,:,1] = 255
            he_[bbox[1]:bbox[3],bbox[0]:bbox[2],:] = aa[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        
    he_[masks==0,:] = 0
    
    show(he_,x.replace('.png','_tunel_like.png'))
    