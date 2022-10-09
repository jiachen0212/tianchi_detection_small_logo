import cv2
import json
import os


val_ann_js = json.load(open('/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/annotations/instances_val2017.json', 'r'))
img_id_names = val_ann_js["images"]
name_id = dict()
id_name = dict()
for dict_ in img_id_names:
    #  "id": 0,
    # "file_name": "1d9efa426d25bd9ff0463b0838f325d0.jpg"
    name_id[dict_["file_name"]] = dict_["id"]
    id_name[dict_["id"]] = dict_["file_name"]

my_res = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/ycy101.json.bbox.json'
res = json.load(open(my_res, 'r'))


name_res = dict()
for dict_ in res:
    if id_name[dict_["image_id"]] not in name_res:
        name_res[id_name[dict_["image_id"]]] = []
#         "image_id": 1,
#         "category_id": 48,
#         "bbox": [
#             246,
#             386,
#             308,
#             71
#         ],
#         "score": 0.9841372966766357
    xyhw = [int(a) for a in dict_["bbox"]]
    xyx1y1 = xyhw[:2] + [xyhw[0]+xyhw[2]] + [xyhw[1]+xyhw[3]]
    line = "{},{},{},{},{},{}".format(xyx1y1[0], xyx1y1[1],xyx1y1[2],xyx1y1[3],dict_["category_id"],dict_["score"])
    name_res[id_name[dict_["image_id"]]].append(line)


val_img_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val/images'
save_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/val_res'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

imgs = os.listdir(val_img_dir)
img_paths = [os.path.join(val_img_dir, a) for a in imgs]

test_ok = open('./ok_test_for_train.txt', 'w')
for img_path in img_paths:
    name = os.path.basename(img_path)

    img = cv2.imread(img_path)
    # point_size = 1
    # thickness = 4
    try:
        pred_res = name_res[name]
    except:
        print("img: {} no result".format(name))
        continue

    for pred in pred_res:
        x1,y1,x2,y2,id_,score = int(pred.split(',')[0]), int(pred.split(',')[1]), int(pred.split(',')[2]), int(pred.split(',')[3]), pred.split(',')[4], pred.split(',')[5]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "cls_id:{}".format(id_), (x2,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 3)
        cv2.putText(img, "score: {}".format(score), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 3)
        # cv2.imshow('1', img)
        # cv2.waitKey(0)
    cv2.imwrite(os.path.join(save_dir, name), img)


    # 0423 交互检测测试数据上模型的表现, ok的可加入train-data做训练数据
    # for pred in pred_res:
    #     x1,y1,x2,y2,id_,score = int(pred.split(',')[0]), int(pred.split(',')[1]), int(pred.split(',')[2]), int(pred.split(',')[3]), pred.split(',')[4], pred.split(',')[5]
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     cv2.putText(img, "cls_id:{}".format(id_), (x2,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 3)
    #     cv2.putText(img, "score: {}".format(score), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 3)
    # cv2.imshow('1', img)
    # while True:
    #     key = cv2.waitKey(0)
    #     if key == ord('q'):
    #         test_ok.write(name+'\n')
    #         print(name)
    #     else:
    #         break


