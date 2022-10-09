import os
import json
import numpy as np

classes = ('冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻')
file = '/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/train/annotations/instances_train2017.json'
data = json.load(open(file, 'r'))
annotations = data["annotations"]

name_id_dict = dict()
id_name_dict = dict()
for dic in data["images"]:
    name_id_dict[dic["file_name"]] = dic["id"]
    id_name_dict[dic["id"]] = dic["file_name"]
with open('./train_name_id.json','w') as fp:
    json.dump(name_id_dict, fp, indent = 4, separators=(',', ': '))
with open('./train_id_name.json','w') as fp:
    json.dump(id_name_dict, fp, indent = 4, separators=(',', ': '))

name_bboxs = dict()
for ann in annotations:
    im_id = int(ann["image_id"])
    if id_name_dict[im_id] not in name_bboxs:
        name_bboxs[id_name_dict[im_id]] = []
    name_bboxs[id_name_dict[im_id]].append(ann["bbox"])


cat_bboxs = dict()
for ann in annotations:
    cat_id = str(ann["category_id"])
    if cat_id not in cat_bboxs:
        cat_bboxs[cat_id] = []
    hw = [int(a) for a in ann["bbox"][2:]]
    cat_bboxs[cat_id].append(hw[0]*hw[1])

cat_areas = dict()
for cat, areas in cat_bboxs.items():
    areas.sort(key=None,reverse=False)
    # 最小的10个box的平均面积
    small_top10_mean = np.mean(areas[:10])
    cat_areas[cat] = small_top10_mean
cat_area = sorted(cat_areas.items(), key=lambda x: x[1], reverse=True)


small_cats = []
for pair in cat_area:
    if pair[1] < 400:
        small_cats.append(classes[int(pair[0])-1])
print(small_cats)