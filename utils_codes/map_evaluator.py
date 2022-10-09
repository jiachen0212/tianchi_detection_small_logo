from SMore_core.evaluation.evaluation_builder import EVALUATORS
from SMore_core.utils.config import merge_dict
from SMore_core.utils.common import (all_gather, distributed, synchronize)

from SMore_det.common.constants import DetectionInputsConstants, DetectionModelOutputConstants
from SMore_det.default_config.evaluation_defaults import EvaluatorDefaults
from SMore_det.evaluation.base_evaluator import DetectionEvaluatorBase
from SMore_det.evaluation.utils.coco_tool import COCOEvalWrapper, COCOWrapper


@EVALUATORS.register_module()
class mAPEvaluator(DetectionEvaluatorBase):
    """
    根据IoU作为评价指标的评测模块。

    Args:
        label_map (list): 包含所有类别的list。
        scores_threshold (float): 预测的boxes计算scores的阈值，如0.35。
        iou_threshold (list): 预测的boxes与gt之间iou的阈值， 如[0.1, 0.2, 0.3]。
    """

    DEFAULT_CONFIG = EvaluatorDefaults.mAPEvaluator_cfg

    def __init__(self, **kwargs):
        self.kwargs = merge_dict(EvaluatorDefaults.mAPEvaluator_cfg, kwargs)
        super().__init__(**kwargs)
        self.proposal_nums = self.kwargs.get('proposal_nums')
        self.label_map = self.kwargs.get('label_map')
        self.classwise = self.kwargs.get('classwise')
        metric = self.kwargs.get('metric')
        self.metric = metric
        allowed_metrics = ['bbox', 'segm']
        if self.metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        self.category_list = [{
            'id': i,
            'name': value
        } for i, value in enumerate(self.label_map)]

        # statistic list
        self.image_list = []
        self.groundtruth_annotations_list = []
        self.detection_list = []

    def process(self, inputs_list, outputs_list, **kwargs):
        for gt, det in zip(inputs_list, outputs_list):
            self.image_list.append(
                {'id': gt[DetectionInputsConstants.IMG_PATH]})
            gt_box_list = gt[DetectionInputsConstants.BBOXES]
            gt_label_list = gt[DetectionInputsConstants.LABELS]
            gt_ignore_box_list = gt.get(DetectionInputsConstants.BBOXES_IGNORE,
                                        [])
            gt_ignore_label_list = gt.get(
                DetectionInputsConstants.LABELS_IGNORE, [])

            gt_box_list = [self.xyxy2xywh(bbox) for bbox in gt_box_list]
            gt_area_list = [self.xywh2area(bbox) for bbox in gt_box_list]
            gt_ignore_box_list = [
                self.xyxy2xywh(bbox) for bbox in gt_ignore_box_list
            ]
            gt_ignore_area_list = [
                self.xywh2area(bbox) for bbox in gt_ignore_box_list
            ]
            for label, bbox, area in zip(gt_label_list, gt_box_list,
                                         gt_area_list):
                self.groundtruth_annotations_list.append({
                    'image_id':
                        gt[DetectionInputsConstants.IMG_PATH],
                    'category_id':
                        label,
                    'bbox':
                        bbox,
                    'area':
                        area,
                    'iscrowd':
                        0
                })
            for label, bbox, area in zip(gt_ignore_label_list,
                                         gt_ignore_box_list,
                                         gt_ignore_area_list):
                self.groundtruth_annotations_list.append({
                    'image_id':
                        gt[DetectionInputsConstants.IMG_PATH],
                    'category_id':
                        label,
                    'bbox':
                        bbox,
                    'area':
                        area,
                    'iscrowd':
                        1
                })

            det_box_list = det[DetectionModelOutputConstants.PREDICT_BOXES]
            det_box_list = [self.xyxy2xywh(bbox) for bbox in det_box_list]
            det_score_list = det[DetectionModelOutputConstants.PREDICT_SCORES]
            det_label_list = det[DetectionModelOutputConstants.PREDICT_LABELS]
            for label, bbox, score in zip(det_label_list, det_box_list,
                                          det_score_list):
                self.detection_list.append({
                    'image_id':
                        gt[DetectionInputsConstants.IMG_PATH],
                    'category_id':
                        label,
                    'bbox':
                        bbox,
                    'score':
                        score
                })

    def evaluate(self):
        if distributed():
            synchronize()
            tmp = all_gather(self.detection_list)
            detection_list = []
            for item in tmp:
                detection_list.extend(item)
            tmp = all_gather(self.groundtruth_annotations_list)
            groundtruth_annotations_list = []
            for item in tmp:
                groundtruth_annotations_list.extend(item)
            tmp = all_gather(self.image_list)
            image_list = []
            for item in tmp:
                image_list.extend(item)
            tmp = all_gather(self.category_list)
            category_list = []
            for item in tmp:
                category_list.extend(item)
        else:
            detection_list = self.detection_list
            groundtruth_annotations_list = self.groundtruth_annotations_list
            image_list = self.image_list
            category_list = self.category_list
        for i, each in enumerate(groundtruth_annotations_list):
            each['id'] = i + 1
        ground_truth_coco = COCOWrapper({
            'annotations': groundtruth_annotations_list,
            'images': image_list,
            'categories': category_list
        })
        detection_coco = ground_truth_coco.LoadAnnotations(detection_list)
        evaluator = COCOEvalWrapper(ground_truth_coco, detection_coco)
        summary_metrics = evaluator.ComputeMetrics(classwise=self.classwise)
        return summary_metrics
