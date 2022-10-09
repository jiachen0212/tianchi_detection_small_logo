import numpy as np
from SMore_core.evaluation.evaluation_builder import EVALUATORS
from SMore_core.utils.config import merge_dict
from SMore_core.utils.common import (all_gather, distributed, synchronize)

from SMore_det.common.constants import DetectionInputsConstants, DetectionModelOutputConstants
from SMore_det.default_config.evaluation_defaults import EvaluatorDefaults
from SMore_det.evaluation.base_evaluator import DetectionEvaluatorBase


@EVALUATORS.register_module()
class IOUEvaluator(DetectionEvaluatorBase):
    """
    根据IoU作为评价指标的评测模块。

    Args:
        label_map (list): 包含所有类别的list。
        scores_threshold (float): 预测的boxes计算scores的阈值，如0.35。
        iou_threshold (list): 预测的boxes与gt之间iou的阈值， 如[0.1, 0.2, 0.3]。
    """

    DEFAULT_CONFIG = EvaluatorDefaults.IoUEvaluator_cfg

    def __init__(self, **kwargs):
        self.kwargs = merge_dict(EvaluatorDefaults.IoUEvaluator_cfg, kwargs)
        super().__init__(**kwargs)
        self.scores_threshold = self.kwargs.get('scores_threshold')
        self.iou_thresholds = self.kwargs.get('iou_threshold')
        self.label_map = self.kwargs.get('label_map')

    def evaluate(self):  # noqa
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
        else:
            detection_list = self.detection_list
            groundtruth_annotations_list = self.groundtruth_annotations_list

        self.all_iou_results = {}  # dict[dict[dict]]  : iou->category->dict
        for i in range(len(self.iou_thresholds)):
            results = {}  # dict[category : dict]
            for j, value in enumerate(self.label_map):
                result_dict = {}
                result_dict['Recall'] = 0
                result_dict['Precision'] = 0
                result_dict['GtNums'] = 0
                result_dict['PNums'] = 0
                result_dict['TP'] = 0
                results[value] = result_dict
            # all_results single iou : dict['all' : dict]
            all_results = {
                'all': {
                    'Recall': 0,
                    'Precision': 0,
                    'GtNums': 0,
                    'PNums': 0,
                    'TP': 0
                }
            }
            results.update(all_results)
            self.all_iou_results[self.iou_thresholds[i]] = results

        self.logger.info('Calculating results for data format ...')
        for det, gt in zip(
                detection_list,  # each image
                groundtruth_annotations_list):
            pred_labels = det[DetectionModelOutputConstants.PREDICT_LABELS]
            pred_boxes = det[DetectionModelOutputConstants.PREDICT_BOXES]
            pred_scores = det[DetectionModelOutputConstants.PREDICT_SCORES]
            keep_index = (pred_scores > self.scores_threshold)
            pred_scores = pred_scores[keep_index]
            pred_labels = pred_labels[keep_index]
            pred_boxes = pred_boxes[keep_index]

            gt_boxes = gt[DetectionInputsConstants.BBOXES]
            gt_labels = gt[DetectionInputsConstants.LABELS]
            num_gt = gt_boxes.shape[0]
            for i, iou_threshold in enumerate(
                    self.iou_thresholds):  # each iou threshold
                have_hitted = []
                for pred_label_id in pred_labels:
                    self.all_iou_results[iou_threshold][
                        self.label_map[pred_label_id]]['PNums'] += 1
                    self.all_iou_results[iou_threshold]['all']['PNums'] += 1
                for gt_label_id in gt_labels:
                    self.all_iou_results[iou_threshold][
                        self.label_map[gt_label_id]]['GtNums'] += 1
                    self.all_iou_results[iou_threshold]['all']['GtNums'] += 1

                for jj in range(num_gt):  # each gt
                    gt_label = gt_labels[jj]
                    gt_box = gt_boxes[jj]
                    x1 = gt_box[0]
                    y1 = gt_box[1]
                    x2 = gt_box[2]
                    y2 = gt_box[3]
                    x_min = np.min((x1, x2))
                    x_max = np.max((x1, x2))
                    y_min = np.min((y1, y2))
                    y_max = np.max((y1, y2))

                    match_index = -1
                    max_iou = 0
                    for pred_i in range(pred_labels.shape[0]):
                        if pred_i not in have_hitted and gt_label == pred_labels[pred_i]:
                            iou = compute_iou([x_min, y_min, x_max, y_max],
                                              pred_boxes[pred_i])
                            if iou > iou_threshold:
                                if iou > max_iou:
                                    max_iou = iou
                                    match_index = pred_i

                    if match_index != -1:
                        self.all_iou_results[iou_threshold][
                            self.label_map[gt_label]]['TP'] += 1
                        self.all_iou_results[iou_threshold]['all']['TP'] += 1
                        have_hitted.append(match_index)

        results = {}
        for per_iou_threshold in self.all_iou_results:  # each iou threshold
            per_iou = self.all_iou_results[per_iou_threshold]
            for r in per_iou.keys():  # each category
                if per_iou[r]['GtNums'] == 0 or per_iou[r]['PNums'] == 0:
                    per_iou[r]['Recall'] = -1
                    per_iou[r]['Precision'] = -1
                else:
                    per_iou[r][
                        'Recall'] = per_iou[r]['TP'] / per_iou[r]['GtNums']
                    per_iou[r][
                        'Precision'] = per_iou[r]['TP'] / per_iou[r]['PNums']
                results[r + '_P'] = per_iou[r]['Precision'] * 100
                results[r + '_R'] = per_iou[r]['Recall'] * 100
            # self.logger.info(
            #     'Evaluation results for bbox(iou_thresholds:{}): \n'.format(
            #         self.iou_thresholds[i]) +
            #     self._create_small_table(results) + '\n')
            results = {}
        print(self.all_iou_results)
        print(list(self.all_iou_results.keys())[0])
        return self.all_iou_results["0.3"]


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0
