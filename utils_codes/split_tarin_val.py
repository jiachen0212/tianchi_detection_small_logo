# coding=utf-8
import json
import os
import random
import copy


file = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/train/annotations/instances_train2017.json'
label = json.load(open(file, 'r'))
# dict_keys(['info', 'license', 'images', 'annotations', 'categories'])
train_js = label.copy()
test_js = label.copy()
categories = label['categories']
license = label['license']
info = label['info']

# need split
imgs = label['images']
annotations = label['annotations']

img_lens = len(imgs)
train_len = int(img_lens*0.7)
random.shuffle(imgs)
train_imgs = imgs[:train_len]
test_imgs = imgs[train_len:]

# img_id和ann的dict对应上
img_id_ann = dict()
for ann in annotations:
    img_id = str(ann["image_id"])
    if img_id not in img_id_ann:
        img_id_ann[img_id] = []
    img_id_ann[img_id].append(ann)

# train
train_anns = []
for img in train_imgs:
    img_id = str(img["id"])
    train_anns.extend(img_id_ann[img_id])

test_anns = []
for img in test_imgs:
    img_id = str(img["id"])
    test_anns.extend(img_id_ann[img_id])

# train_js
train_js['images'] = train_imgs
train_js['annotations'] = train_anns
with open('./train.json' , "w", encoding='utf-8') as fp:
    json.dump(train_js, fp, ensure_ascii=False,indent = 4)

# test_js
test_js['images'] = test_imgs
test_js['annotations'] = test_anns
with open('./test.json' , "w", encoding='utf-8') as fp:
    json.dump(test_js, fp, ensure_ascii=False,indent = 4)