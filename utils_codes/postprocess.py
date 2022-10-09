# coindg=utf-8
import json
import collections

res_name = '2.json'
js_name = 'ohem1.json.bbox.json'

ress = json.load(open(js_name, 'r'))

# out = []
# for res in ress:
#     tmp = dict()
#     if res["score"] >= 0.3:
#         tmp["bbox"] = [int(a) for a in res["bbox"]]
#         tmp["image_id"] = res["image_id"]
#         tmp["category_id"] = res["category_id"]
#         tmp["score"] = res["score"]
#         out.append(tmp)
#     else:
#         print(res)
# with open('./ycycas.json' , "w", \ncoding='utf-8') as fp:
#     json.dump(out, fp, ensure_ascii=False,indent = 4)


# postprocess
# 1. 一张图检出类别一致性这个后处理还是要的
# 2. score卡个阈值, 0.1, 0.3之类
# 3. 结果json画图可视化下看看效果再提交


def max_num_category_id(k, v):
    img_res = []
    category_ids = [a["category_id"] for a in v]
    scores = [a["score"] for a in v]
    bboxs = [a["bbox"] for a in v]
    caid_score = dict()
    caid_bbox = dict()
    for ind, ca_id in enumerate(category_ids):
        if str(ca_id) not in caid_score:
            caid_score[str(ca_id)] = []
            caid_bbox[str(ca_id)] = []
        caid_score[str(ca_id)].append(scores[ind])
        caid_bbox[str(ca_id)].append(bboxs[ind])

    res = collections.Counter(category_ids)
    res_dict = dict(res)
    if len(res_dict) > 1:
        res_ = res.most_common(2)
        if res_[0][1] == res_[1][1]:
            score1 = max(caid_score[str(res_[0][0])])
            score2 = max(caid_score[str(res_[1][0])])
            if score1 >= score2:
                for ind, score in enumerate(caid_score[str(res_[0][0])]):
                    tmp = dict()
                    tmp["category_id"] = res_[0][0]
                    tmp["score"] = score
                    tmp["bbox"] = [int(a) for a in caid_bbox[str(res_[0][0])][ind]]
                    tmp["image_id"] = k
                    img_res.append(tmp)
            else:
                for ind, score in enumerate(caid_score[str(res_[1][0])]):
                    tmp = dict()
                    tmp["category_id"] = res_[1][0]
                    tmp["score"] = score
                    tmp["bbox"] = [int(a) for a in caid_bbox[str(res_[1][0])][ind]]
                    tmp["image_id"] = k
                    img_res.append(tmp)
        else:
            for ind, score in enumerate(caid_score[str(res_[0][0])]):
                tmp = dict()
                tmp["category_id"] = res_[0][0]
                tmp["score"] = score
                tmp["bbox"] = [int(a) for a in caid_bbox[str(res_[0][0])][ind]]
                tmp["image_id"] = k
                img_res.append(tmp)
        return img_res
    else:
        for a in v:
            a["bbox"] = [int(c) for c in a["bbox"]]
        return v


img_ress = dict()
for res in ress:
    img_id = res["image_id"]
    if img_id not in img_ress:
        img_ress[img_id] = []
    img_ress[img_id].append(res)

all_ = []
for k, v in img_ress.items():
    res = max_num_category_id(k, v)
    all_.extend(res)

ress = []
for a in all_:

    # if a["score"] >= 0.3:
    #     # 将score映射到: [0.5,1]
    #     score = a["score"]/2+0.5
    #     a["score"] = score
    #     ress.append(a)
    # else:
    #     print(a)

    # 将score映射到: [0.5,1]
    score = a["score"]/2+0.5
    a["score"] = score
    ress.append(a)

with open(res_name , "w", encoding='utf-8') as fp:
    json.dump(ress, fp, ensure_ascii=False,indent = 4)
