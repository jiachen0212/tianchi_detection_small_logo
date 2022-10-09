import os
import json
import numpy as np
import pandas as pd
import cv2
import glob
import random
from PIL import Image
import time
from sklearn import metrics as mr


def generate_duck_json():
    random.seed(666)

    aug_name=['Sanyo/三洋', 'erke/鸿星尔克', 'OSPREY', 'PSALTER/诗篇', 'BOY LONDON', 'Granite Gear', 'Beaster', 'PALLADIUM', 'KENZO', 'JORDAN', 'Gucci/古驰', 'Tory Burch/汤丽柏琦', 'ON/昂跑', 'Swatch/斯沃琪', 'Lining/李宁', 'TREYO/雀友', 'PINKO', 'Clarins/娇韵诗', 'GXG', 'UNDEFEATED', 'Mizuno/美津浓', 'Trendiano', 'carhartt', 'YONEX/尤尼克斯', 'DESCENTE/迪桑特']
    defect_img_root='/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/train/images/'
    classes = ('冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻')
    save_dir='./normal_aug/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    anno_file='/Users/chenjia/Downloads/Learning/SmartMore/2022/TIANCHI/小log检测/train/annotations/instances_train2017.json'
    anno_data= json.load(open(anno_file,"r"))
    annotations = anno_data["annotations"]
    id_name = json.load(open('./train_id_name.json', 'r'))
    imname_bboxs = dict()
    imname_cats = dict()
    for ann in annotations:
        im_id = ann["image_id"]
        im_name = id_name[str(im_id)]
        if im_name not in imname_bboxs:
            imname_bboxs[im_name] = []
            imname_cats[im_name] = []
        imname_bboxs[im_name].append(ann["bbox"])
        imname_cats[im_name].append(classes[int(ann["category_id"])-1])

    ring_width=10 # default is 5

    result=[]
    last_result_length=0
    img_name_count=0
    train_imgs = [a for a in os.listdir(defect_img_root) if '.jpg' in a]

    for img_name in train_imgs:
        bboxs = imname_bboxs[img_name]
        img_name_count+=1
        defect_names = imname_cats[img_name]
        testimg=Image.open(os.path.join(defect_img_root, img_name))
        temp_img=Image.open(os.path.join(defect_img_root, img_name))
        save_temp_name='template_'+img_name
        box_len = len(bboxs)
        for idx in range(box_len):
            pts=bboxs[idx]
            d_name=defect_names[idx]
            xmin=pts[0]
            ymin=pts[1]
            xmax=pts[2]+pts[0]
            ymax=pts[3]+pts[1]
            defect_h=pts[2]
            defect_w=pts[3]
            w_h=round(defect_w/defect_h,2)
            h_w=round(defect_h/defect_w,2)
            if w_h > 5 or h_w >5 or (ymax-ymin)*(xmax-xmin)>300000:
                left_top_x=random.randint(1,3)
                left_top_y=random.randint(1,testimg.size[1])
            else:
                left_top_x=random.randint(1,testimg.size[0])
                left_top_y=random.randint(1,testimg.size[1])
            mask=np.zeros_like(temp_img)

            if d_name in aug_name:
                scale=random.randint(3,5)
                mask[int(scale*(left_top_y-ring_width)):int(scale*(left_top_y+defect_w+ring_width)),int(scale*(left_top_x-ring_width)):int(scale*(left_top_x+defect_h+ring_width))]=255
                mask[int(left_top_y):int(scale*(left_top_y+defect_w)),int(left_top_x):int(scale*(left_top_x+defect_h))]=0
                patch=testimg.crop((xmin,ymin,xmax,ymax))
                patch1=patch.copy()
                try:
                    patch2=temp_img.crop((left_top_x,left_top_y,int(left_top_x+patch1.size[0]),int(left_top_y+patch1.size[1])))

                except:
                    continue
                patch2.resize((patch1.size[0],patch1.size[1]))
                patch1=np.resize(patch1,-1)
                patch2=np.resize(patch2,-1)

                try:
                    mutual_infor=mr.mutual_info_score(patch1,patch2)
                except:
                    continue

                if mutual_infor>0.8:
                    patch=patch.resize((patch.size[0]*scale,patch.size[1]*scale))
                    temp_img.paste(patch,(left_top_x,left_top_y))
                    temp_img = cv2.cvtColor(np.asarray(temp_img),cv2.COLOR_RGB2BGR)
                    try:
                        temp_img = cv2.inpaint(temp_img,mask[:,:,0],3,cv2.INPAINT_TELEA)
                    except:
                        continue
                    temp_img = Image.fromarray(cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB))
                    result.append({'name': save_temp_name, 'defect_name': d_name, 'bbox': [left_top_x,left_top_y,left_top_x+defect_h*scale,left_top_y+defect_w*scale]})

                else:
                    continue
            else:
                mask[int(left_top_y-ring_width):int(left_top_y+defect_w+ring_width),int(left_top_x-ring_width):int(left_top_x+defect_h+ring_width)]=255
                mask[int(left_top_y):int(left_top_y+defect_w),int(left_top_x):int(left_top_x+defect_h)]=0

                patch=testimg.crop((xmin,ymin,xmax,ymax))

                patch1=patch.copy()

                patch1=patch.copy()
                try:
                    patch2=temp_img.crop((left_top_x,left_top_y,int(left_top_x+patch1.size[0]),int(left_top_y+patch1.size[1])))

                except:
                    continue
                patch2.resize((patch1.size[0],patch1.size[1]))
                patch1=np.resize(patch1,-1)
                patch2=np.resize(patch2,-1)
                try:
                    mutual_infor=mr.mutual_info_score(patch1,patch2)
                except:
                    continue

                #==================================================================================================#
                if mutual_infor>0.8:
                    temp_img.paste(patch,(left_top_x,left_top_y))
                    temp_img = cv2.cvtColor(np.asarray(temp_img),cv2.COLOR_RGB2BGR)
                    try:
                        temp_img = cv2.inpaint(temp_img,mask[:,:,0],3,cv2.INPAINT_TELEA)
                    except:
                        continue
                    temp_img = Image.fromarray(cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB))
                    result.append({'name': save_temp_name, 'defect_name': d_name, 'bbox': [left_top_x,left_top_y,left_top_x+defect_h,left_top_y+defect_w]})

                else:
                    continue
        try:
            temp_img.save(os.path.join(save_dir, save_temp_name))
        except:
            continue

        #test path
        json_name='./Duck_inject_normal.json'
        with open(json_name,'w') as fp:
            json.dump(result, fp, indent = 4, separators=(',', ': '))




classes = ('冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻')
file = './Duck_inject_normal.json'
data = json.load(open(file, 'r'))

name_anns = dict()
name_cat = dict()
for da in data:
    name = da["name"]
    name_cat[name] = classes.index(da["defect_name"])+1
    bbox = [int(a) for a in da["bbox"]]
    bbox = bbox[:2] + [bbox[2]-bbox[0]] + [bbox[3]-bbox[1]]
    if name not in name_anns:
        name_anns[name] = []
    name_anns[name].append(bbox)


# imname_cat = dict()
# for d in data:
#     if d["name"] not in imname_cat:
#         imname_cat[d["name"]] = []
#     imname_cat[d["name"]].append(d["defect_name"])

# for im, v in imname_cat.items():
#     a = []
#     for b in v:
#         if b not in a:
#             a.append(b)
#     assert len(a) == 1

base_ann = json.load(open('./小log检测/train/annotations/instances_train2017.json', 'r'))
# _keys(['info', 'license', 'images', 'annotations', 'categories'])

all_images = base_ann["images"]
all_anns = base_ann["annotations"]
au_imgs = './normal_aug'
count = 3000
ann_cnt = 6666

for name, anns in name_anns.items():

    # img
    image = cv2.imread(os.path.join(au_imgs, name))
    h, w = image.shape[0], image.shape[1]
    tmp = dict()
    tmp["height"] = h
    tmp["width"] = w
    tmp["id"] = count
    tmp["file_name"] = name
    all_images.append(tmp)

    # ann
    for ann in anns:
        ttmp = dict()
        ttmp["id"] = ann_cnt
        ann_cnt += 1
        ttmp["image_id"] = count
        ttmp["bbox"] = ann
        ttmp["iscrowd"] = 0
        ttmp["area"] = ann[2]*ann[3]
        ttmp["segmentation"] = [[]]
        ttmp["category_id"] = name_cat[name]
        all_anns.append(ttmp)
    count += 1

base_ann["images"] = all_images
base_ann["annotations"] = all_anns

json_name='./added_Duck_inject.json'
with open(json_name,'w') as fp:
    json.dump(base_ann, fp, indent = 4, separators=(',', ': '))

