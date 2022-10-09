# coding=utf-8
import json
import random
import os
# 生成结果提交json

file = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/annotations/instances_val2017.json'
res = json.load(open(file, 'r'))
img_id_name = dict()
for img in res['images']:
    img_id_name[img["id"]] = img["file_name"]

base_json_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测'
js_path = os.path.join(base_json_path, 'chenjiamm33.json')
model_res = json.load(open(js_path, 'r'))

file = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/annotations/instances_val2017.json'
res = json.load(open(file, 'r'))
res_list = []
for img in res['images']:
    # 一张图片里可能检出多个bbox
    for line in model_res[img_id_name[img["id"]]]:
        single_res = dict()
        x, y, x2, y2, score, pre_ind = line.split(',')
        h, w = int(x2)-int(x), int(y2) - int(y)
        if float(score) >= 0.1:
            single_res["image_id"] = int(img["id"])
            single_res["category_id"] = int(pre_ind)
            single_res["bbox"] = [int(a) for a in [x, y, h, w]]
            single_res["score"] = float(score)
            res_list.append(single_res)

with open("./chenjiaycy.json" , "w", encoding='utf-8') as fp:
    json.dump(res_list, fp, ensure_ascii=False,indent = 4)
