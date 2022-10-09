# coding=utf-8
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import os
import json


config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_26.pth'

# init model
model = init_detector(config_file, checkpoint_file)

# test single_img
# score_thres = 0.1
# img = './test.jpg'
# image = cv2.imread(img)
# result = inference_detector(model, img)
# result = [a for a in result if len(a) > 0]
# all_res = []
# for res in result:
#     all_res.extend(res)
# for res in all_res:
#     if res[-1] >= score_thres:
#         bbox = [int(a) for a in res[:4]]
#         image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
# cv2.imwrite('./res.jpg', image)


# test val-data
val_res = dict()
score_thres = 0.1
test_path = '/data/home/jiachen/data/tc_cc33/logo/val/images'
tests = os.listdir(test_path)
test_imgs = [os.path.join(test_path, a) for a in tests]
for test_img in test_imgs:
    basename = os.path.basename(test_img)
    image = cv2.imread(test_img)
    try:
        image.shape
    except:
        continue
    result = inference_detector(model, image)
    single_res = []
    score_max_list = []
    for res in result:
        if len(res) > 0:
            res_score = [a[-1] for a in res]
            score_max_list.append(max(res_score))
        else:
            score_max_list.append(0)
    assert len(score_max_list) == 50
    cls_ind = score_max_list.index(max(score_max_list))
    res = result[cls_ind]
    for single_pred in res:
        score = single_pred[-1]
        if score >= score_thres:
            bbox = [int(a) for a in single_pred[:4]]
            # image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # single_res.append(bbox + [single_pred[-1]] + [cls_ind])
            res_line = "{},{},{},{},{},{}".format(bbox[0], bbox[1], bbox[2], bbox[3], score, cls_ind)
            single_res.append(res_line)
            print("img: {}, res: {}".format(basename, res_line))
    # print("img: {}, res: {}".format(basename, single_res))
    val_res[basename] = single_res

with open("./chenjiamm1.json" , "w", encoding='utf-8') as fp:
    json.dump(val_res, fp, ensure_ascii=False,indent = 4)


