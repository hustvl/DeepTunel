from .evaluator import DatasetEvaluator
from .registry import EVALUATOR
from cvpods.utils import comm, PathManager
import itertools
from collections import OrderedDict
import torch
import logging
import os
import numpy as np




def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious



@EVALUATOR.register()
class MidogEvaluator(DatasetEvaluator):
    def __init__(
        self,
        iou_thrs=0.5,
        score_thrs=0,
        metric="F1score",
        distributed=True,
        output_dir=None,
    ):
        self._distributed = distributed
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir
        self.iou_thrs = iou_thrs
        self.score_thrs = score_thrs
        self.metric = metric
        self._cpu_device = torch.device("cpu")
    def reset(self):
        self._predictions = []
    
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"file_name": input["file_name"]}

            if "instances" in input:
                instances = input["instances"].to(self._cpu_device)
                prediction["gt_instances"] = instances

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            
            self._predictions.append(prediction)

    
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
        
        if len(predictions) == 0:
            self._logger.warning("[MidogEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = self.evaluate_metric(predictions,self.metric,self.iou_thrs,self.score_thrs)
        
        return self._results

    def evaluate_metric(self,
                 results,
                 metric='F1score',
                 iou_thrs=0.5,
                 score_thrs=0):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['precision', 'recall', 'F1score']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        annotations = [result['gt_instances'] for result in results]
        predictions = [result['instances'] for result in results]
        predictions_class = [np.array(x.pred_classes) for x in predictions]
        predictions_bboxes = [np.array(torch.hstack([x.pred_boxes.tensor,x.scores.reshape(-1,1)]))[class_i==0] \
                                    for x,class_i in zip(predictions,predictions_class)]
        score_thrs = [0,0.05,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,0.95]
        if isinstance(iou_thrs,float):
            iou_thrs = [iou_thrs]
        else:
            assert isinstance(iou_thrs,list)
        if isinstance(score_thrs,float):
            score_thrs = [score_thrs]
        else:
            assert isinstance(score_thrs,list)
        precisions = []
        recalls = []
        f1scores = []
        for score_thr in score_thrs:
            new_predictions_bboxes = []
            for re in predictions_bboxes:
                score_valid = re[:,-1] > score_thr
                re = re[score_valid]
                new_predictions_bboxes.append(re)
            for iou_thr in iou_thrs:
                fps = []
                tps = []
                gts = []
                assert len(annotations) == len(new_predictions_bboxes)
                for idx, (pred_bbox, annotation) in enumerate(zip(new_predictions_bboxes, annotations)):
                    fp = np.zeros(len(pred_bbox),dtype=np.int32)
                    tp = np.zeros(len(pred_bbox),dtype=np.int32)

                    # if there is no gt bboxes in this image, then all det bboxes
                    # within area range are false positives
                    if len(annotation)==0:
                        fp[...] = 1
                        num_gts = 0
                    else:
                        gt_bboxes = np.array(annotation.gt_boxes.tensor)
                        num_gts = len(gt_bboxes)
                        assert num_gts > 0
                        if len(pred_bbox)>0:
                            det_bboxes = pred_bbox
                            ious = bbox_overlaps(
                                    det_bboxes[:, :4],
                                    gt_bboxes,
                                    )
                            # for each det, the max iou with all gts
                            ious_max = ious.max(axis=1)
                            # for each det, which gt overlaps most with it
                            ious_argmax = ious.argmax(axis=1)
                            # sort all dets in descending order by scores
                            sort_inds = np.argsort(-det_bboxes[:, -1])
                            gt_covered = np.zeros(num_gts, dtype=bool)
                            for i in sort_inds:
                                if ious_max[i] >= iou_thr:
                                    matched_gt = ious_argmax[i]
                                    if not gt_covered[matched_gt]:
                                        gt_covered[matched_gt] = True
                                        tp[i] = 1
                                    else:
                                        fp[i] = 1
                                # otherwise ignore this detected bbox, tp = 0, fp = 0
                                else:
                                    fp[i] = 1
                    fps.append(fp)
                    tps.append(tp)          
                    gts.append(num_gts)
                TP = sum([len(np.where(tp==1)[0]) for tp in tps])
                FP = sum([len(np.where(fp==1)[0]) for fp in fps])
                GT = sum(gts)
                eps = np.finfo(np.float32).eps
                precision = TP / (TP+FP+eps)
                recall = TP / (GT + eps)
                precisions.append(precision)
                recalls.append(recall)
                f1scores.append((2*precision * recall) / (precision + recall))

        return {"precision":precisions,"recall":recalls,"f1score":f1scores}