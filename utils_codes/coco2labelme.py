import json
import os
import cv2
import numpy as np


if __name__ == "__main__":

    file = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/train/annotations/instances_train2017.json'
    label = json.load(open(file, 'r'))
    imgs = label['images']
    annotations = label['annotations']
    categories = label['categories']
    categories = [a["name"] for a in categories]
    print(categories)
    json_save_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/train/images'

    # 对annotations做以img_id为key的dict预处理, value是bbox两个点信息.
    img_id_anns = dict()
    for dic in annotations:
        img_id = dic["image_id"]
        categorie_id = dic["category_id"]
        if img_id not in img_id_anns:
            img_id_anns[img_id] = []
        img_id_anns[img_id].append([dic["bbox"], categories[categorie_id-1]])

    # for k, v in img_id_anns.items():
    #     print(k, v)

    for img_dict in imgs:
        img_name = img_dict["file_name"]
        lab_dict = {"version": "4.5.10", "flags": {}, "shapes": [], "imageData": None}

        # 填入infos
        id_ = img_dict["id"]
        lab_dict["imagePath"] = img_name
        for box_name in img_id_anns[id_]:
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
        with open(os.path.join(json_save_dir, "{}.json".format(img_name.split('.')[0])) , "w", encoding='utf-8') as fp:
            json.dump(lab_dict, fp, ensure_ascii=False,indent = 4)



    # # boox check
    # img_path = '/Users/chenjia/Desktop/check_json/0a4f4202d6fa86f766b0c6de7bc6582c.jpg'
    # img = cv2.imread(img_path)
    # point_size = 1
    # thickness = 4
    # # coors = [[326, 63], [39, 14], [308, 108], [125, 35]]
    # coors = [[326, 63], [365, 77], [308, 108], [433, 143]]
    # colors = [(255, 0, 255), (0,255,0), (0,0,255), (255,255,255)]
    # for ind, coor in enumerate(coors):
    #     cv2.circle(img, (int(coor[0]),int(coor[1])), point_size, colors[ind], thickness)
    # cv2.imshow('1', img)
    # cv2.waitKey(0)












