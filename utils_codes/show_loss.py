# coding=utf-8

# lines = open('./output/train-2022-04-21@04:16:47.log').readlines()
# for line in lines:
#     if 'CrossEntropyLoss_0:' in line:
#         iter_ = line.split('iter: ')[1][:6]
#         lr_ = line.split('lr: ')[1][:6]
#         fl_loss = line.split('FocalLoss_0: ')[1][:6]
#         iou_loss = line.split('IoULoss_0: ')[1][:6]
#         ce_loss = line.split('CrossEntropyLoss_0: ')[1][:6]
#         print('iter: {}, lr: {}, fl_loss: {}, iou_loss: {}, ce_loss: {}'.format(iter_, lr_, fl_loss, iou_loss, ce_loss))


import os
import matplotlib.pyplot as plt


# files = os.listdir('./output')
# files = [a for a in files if a.split('.')[-1] == 'log']
# lines = open('./output/{}'.format(files[0])).readlines()
# for line in lines:
#     if "lr: " in line:
#         epoch = line.split('Epoch ')[1][:7]
#         lr_ = line.split('lr: ')[1][:9]
#         loss_rpn_cls = line.split('loss_rpn_cls: ')[1][:6]
#         loss_rpn_bbox = line.split('loss_rpn_bbox: ')[1][:6]
#         loss_cls = line.split('loss_cls: ')[1][:6]
#         loss_bbox = line.split('loss_bbox: ')[1][:6]
#         loss = line.split('loss: ')[1][:6]
#         print('epoch: {}, lr: {}, loss_rpn_cls: {}, loss_rpn_bbox: {}, loss_cls: {}, loss_bbox: {}, loss: {}'.format(epoch, lr_, loss_rpn_cls, loss_rpn_bbox, loss_cls, loss_bbox, loss))




# 2022-04-26 03:11:29,533 - mmdet - INFO - Epoch [57][1700/1721]  lr: 1.000e-05, eta: 0:41:27, time: 0.480, data_time: 0.013, memory: 6686, loss_rpn_cls: 0.0058, loss_rpn_bbox: 0.0040, s0.loss_cls: 0.0662, s0.acc: 97.8281, s0.loss_bbox: 0.0251, s1.loss_cls: 0.0330, s1.acc: 97.6218, s1.loss_bbox: 0.0286, s2.loss_cls: 0.0173, s2.acc: 97.7014, s2.loss_bbox: 0.0238, loss: 0.2039

cls0, cls1, cls2, acc0, acc1, acc2, losses = [], [], [], [], [], [], []
clss, box_loss = [], []

files = ['20220510_034237.log']
lines = open(files[0]).readlines()
for line in lines:
    if "lr: " in line:
        epoch = line.split('Epoch ')[1][:7]
        lr_ = line.split('lr: ')[1][:9]

        # swin
        # loss = line.split('loss: ')[1][:6]
        # loss_cls = line.split('loss_cls: ')[1][:6]
        # loss_bbox = line.split('loss_bbox: ')[1][:6]
        # clss.append(float(loss_cls))
        # box_loss.append(float(loss_bbox))
        # losses.append(float(loss))

        
        # cascade
        loss = line.split('loss: ')[1][:6]     

        loss_rpn_bbox = line.split('loss_rpn_bbox: ')[1][:6]
        loss_rpn_cls = line.split('loss_rpn_cls: ')[1][:6]
        
        s0loss_cls = line.split('s0.loss_cls: ')[1][:6]
        s0acc = line.split('s0.acc: ')[1][:6]

        s1loss_cls = line.split('s1.loss_cls: ')[1][:6]
        s1acc = line.split('s1.acc: ')[1][:6]

        s2loss_cls = line.split('s2.loss_cls: ')[1][:6]
        s2acc = line.split('s2.acc: ')[1][:6]

        print('epoch: {}, lr: {}, loss_rpn_bbox: {}, loss_rpn_cls: {}, s0.loss_cls: {}, s0.acc: {}, s1.loss_cls: {}, s1.acc: {}, s2.loss_cls: {}, s2.acc: {}, loss: {}'.format(epoch, lr_, loss_rpn_bbox, loss_rpn_cls, s0loss_cls, s0acc, s1loss_cls, s1acc, s2loss_cls, s2acc, loss))
        cls0.append(float(s0loss_cls))
        cls1.append(float(s1loss_cls))
        cls2.append(float(s2loss_cls))
        acc0.append(float(s0acc))
        acc1.append(float(s1acc))
        acc2.append(float(s2acc))
        losses.append(float(loss))

lens = len(losses)
# x = [int(a*50/311) for a in range(lens)]
x = [a*50 for a in range(lens)]
plt.plot(x,losses)
plt.show()

