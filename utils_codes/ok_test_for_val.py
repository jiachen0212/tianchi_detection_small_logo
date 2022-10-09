# coding=utf-8
import os
import json
import cv2

# all_val = json.load(open('/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/annotations/instances_val2017.json', 'r'))
# vals = all_val["images"]
# name_id = dict()
# for val in vals:
#     name_id[val["file_name"]] = val["id"]
# with open('./name_id.json' , "w", encoding='utf-8') as fp:
#     json.dump(name_id, fp, ensure_ascii=False,indent = 4)


name_id = json.load(open('./name_id.json', 'r'))
lines = open('./val.txt', 'r').readlines()
imgs = []
for line in lines:
    imgs.append(line[:-1])

test_res = json.load(open('./ccc.json', 'r'))
id_res = dict()
for res in test_res:
    if res["image_id"] not in id_res:
        id_res[res["image_id"]] = []
    id_res[res["image_id"]].append(res["bbox"] + [res["category_id"]])
with open('./id_res.json' , "w", encoding='utf-8') as fp:
    json.dump(id_res, fp, ensure_ascii=False,indent = 4)

val_img_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/images'
all_val = json.load(open('/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/annotations/instances_val2017.json', 'r'))
train_ok_for_val = all_val.copy()
train_ok_for_val["images"] = []
train_ok_for_val["annotations"] = []
ann_id = 0
for ind, im in enumerate(imgs):
    image = cv2.imread(os.path.join(val_img_path, im))
    h, w = image.shape[0], image.shape[1]
    tmp = dict()
    tmp["height"] = h
    tmp["width"] = w
    tmp["id"] = ind
    tmp["file_name"] = im
    train_ok_for_val["images"].append(tmp)

    # 根据img_name + id_res, 把检测结果取出来
    try:
        bboxs_catids = id_res[name_id[im]]
    except:
        continue
    for bboxs_catid in bboxs_catids:
        ann = dict()
        ann["image_id"] = ind
        ann["id"] = ann_id
        ann_id += 1
        ann["category_id"] = bboxs_catid[-1]
        ann["bbox"] = bboxs_catid[:-1]
        ann["iscrowd"] = 0
        ann["area"] = bboxs_catid[2]*bboxs_catid[3]
        ann["segmentation"] = [[]]
        train_ok_for_val["annotations"].append(ann)

with open('./trainok_forval.json' , "w", encoding='utf-8') as fp:
    json.dump(train_ok_for_val, fp, ensure_ascii=False,indent = 4)