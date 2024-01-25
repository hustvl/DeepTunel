import torch
from torch import nn
import torchvision.ops as ops 
from cvpods.structures import Boxes, ImageList, Instances
import os.path as osp
import os
from cvpods.modeling.losses import SigmoidFocalLoss
from sklearn.metrics import precision_score, recall_score
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.final = nn.Linear(nb_filter[0], 1)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # if config['loss'] == 'BCEWithLogitsLoss':
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        # self.criterion = SigmoidFocalLoss(alpha=0.8,gamma=0.5)
        # self.train_metic = {"precision":AverageMeter(),
        #                     "recall":AverageMeter(),
        #                     "acc":AverageMeter(),
        #                     "f1score":AverageMeter(),
        #                     }
        self.to(self.device)
        # else:
            # criterion = losses.__dict__[config['loss']]().cuda()

    def convert_to_roi_format(self, boxes, length):
        rois = []
        for i in range(len(boxes)):
            # boxes[i][:,2] = boxes[i][:,2] + boxes[i][:,0]
            # boxes[i][:,3] = boxes[i][:,3] + boxes[i][:,1]
            rois.append(boxes[i].tensor[:length[i]].float())
        return rois

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,0)
        return images
    
    def random_choice(self, gallery, num):
    # def random_choice( gallery, num):
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def rois_sample(self,rois,target:torch.Tensor,ratio=1):

        rois_ = torch.cat([torch.hstack([roi,torch.ones(roi.shape[0],1,device=roi.device) * i]) for i,roi in enumerate(rois)])
        num_neg = torch.where(target == 0)[0].numel()
        num_pos = len(target) - num_neg
        pos_ind = torch.where(target == 1)[0]
        neg_ind = torch.where(target == 0)[0]
        num_b = min(num_neg,num_pos)
        
        if num_pos > num_b:
            pos_ind_sample = self.random_choice(pos_ind, num_b)
            pos_ind_sample = pos_ind_sample.unique()
            pos_ind_sample.sort()
        else:
            pos_ind_sample = pos_ind

        if num_neg > num_b:
            neg_ind_sample = self.random_choice(neg_ind, num_b)
            neg_ind_sample = neg_ind_sample.unique()
            neg_ind_sample.sort()
        else:
            neg_ind_sample = neg_ind
        
        ret_rois = torch.vstack([rois_[pos_ind_sample],rois_[neg_ind_sample]])

        return [ret_rois[ret_rois[:,-1]==i][:,:-1] for i in range(len(rois))]

    def forward(self, batched_inputs):
    # def forward(self, input, bboxes, length):
        # from pdb import set_trace
        # set_trace()
        # for x in batched_inputs:
            # gt_img = osp.join(osp.basename(x['file_name']))
        images = self.preprocess_image(batched_inputs)
        gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ] 
        length = [
                x["length"].to(self.device) for x in batched_inputs
            ] 

        bboxes = [x.gt_boxes for x in gt_instances]
        labels = [x.gt_classes for x in gt_instances]

        x0_0 = self.conv0_0(images.tensor)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))


        rois = self.convert_to_roi_format(bboxes, length)
        if self.training:
            target = []
            for batch_i in range(len(labels)):
                target.append(labels[batch_i][:length[batch_i]])
            target = torch.cat(target)


        roi_feas = ops.roi_pool(x0_4, rois, (7,7), spatial_scale=1.0)
        x = self.avgpool(roi_feas)
        x = x.view(x.size(0), -1)
        if x.size(0) % 32 != 0:
            padding = torch.zeros((x.size(0) % 32,x.size(1))).to(x.device)
            x = torch.cat((x,padding))

        output = self.final(x)
        # if self.training:
        #     output_clone = np.array((output[:len(target)].detach().clone().sigmoid().cpu() > 0.5).int())
        #     target_cpu = np.array(target.unsqueeze(1).int().cpu())
        #     recall = recall_score(target_cpu, output_clone, average='binary')
        #     precision = precision_score(target_cpu, output_clone, average='binary')
        #     acc = (output_clone==target_cpu).sum() / float(len(target_cpu))
        #     f1score = 2*recall*precision /(recall+precision+1e-6)
            
        #     self.train_metic['precision'].update(precision)
        #     self.train_metic['recall'].update(recall)
        #     self.train_metic['acc'].update(acc)
        #     self.train_metic['f1score'].update(f1score)

        if self.training:
            loss = self.criterion(output[:len(target)], target.unsqueeze(1).float())
            if torch.isnan(loss) :
                from pdb import set_trace
                set_trace()
            return {'loss':loss}
        else:
            return output[:len(labels[0])].sigmoid()
            # return (output[:len(labels[0])].sigmoid() > 0.5).int().squeeze(-1)
