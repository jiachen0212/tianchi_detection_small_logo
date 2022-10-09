# coding=utf-8
import os
import json

pre = 'jiachen/data/tc_cc33/logo/train/images'
dir_ = '/data/home/jiachen/data/tc_cc33/logo/val/images'
imgs = os.listdir(dir_)
test_ = open('./test.txt', 'w')
for img in imgs:
    line = '{}||{}\n'.format(os.path.join(pre, img), os.path.join(pre, img.split('.')[0]+'.json'))
    print(line)
    test_.write(line)

    lab_dict = {"version": "4.5.10", "flags": {}, "shapes": [], "imageData": None}
    id_ = img_dict["id"]
    lab_dict["imagePath"] = img

    tmp = {"group_id": None, "shape_type": "rectangle", "flags": {}}

    tmp["label"] = box_name[1]
    point1 = box_name[0][:2]
    h, w = box_name[0][2:][0], box_name[0][2:][1]
    point2 = [point1[0]+h, point1[1]+w]
    tmp["points"] = [point1, point2]
    lab_dict["shapes"].append(tmp)
    lab_dict["imageHeight"] = img_dict["height"]
    lab_dict["imageWidth"] = img_dict["width"]
    lab_dict["id"] = id_
    with open(os.path.join(json_save_dir, "{}.json".format(img.split('.')[0])) , "w", encoding='utf-8') as fp:
        json.dump(lab_dict, fp, ensure_ascii=False,indent = 4)





