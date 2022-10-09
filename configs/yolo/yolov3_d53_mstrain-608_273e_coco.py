_base_ = '../_base_/default_runtime.py'
dataset_type = 'CocoDataset'
data_root = '/data/home/jiachen/data/tc_cc33/logo/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (768, 768)  # height, width


albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    # bug... 不能用.
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # bug... 不能用.
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    # dict(type='Resize', img_scale=[(400, 400), (800, 800)], keep_ratio=True),
    dict(type='Resize', img_scale=[(800, 800), (600, 600)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# model settings
classes = ('冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻')
# model settings
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=50,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(336, 220), (208, 55), (170, 123)], 
                        [(102, 35), (91, 83), (51, 56)], 
                        [(20, 13), (29, 32), (52, 19)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 1333),
        # img_scale=[(800, 800), (600, 600)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# data = dict(
#     samples_per_gpu=6,
#     workers_per_gpu=4, 
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'train/labelme2coco/train_yolo.json',
#         img_prefix=data_root + 'train/images/',
#         classes=classes,
#         pipeline=train_pipeline),
#     val=dict( 
#         type=dataset_type,
#         ann_file=data_root + 'train/labelme2coco/val_yolo.json',
#         img_prefix=data_root + 'val/images/',
#         classes=classes,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'val/annotations/instances_val2017.json',
#         img_prefix=data_root + 'val/images/',
#         classes=classes,
#         pipeline=test_pipeline))
data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'train/labelme2coco/train.json',
        ann_file=data_root + 'train/annotations/instances_train2017.json',
        img_prefix=data_root + 'train/images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelme2coco/trainok_forval.json',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotations/instances_val2017.json',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,  # 2000 same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=273)
# evaluation = dict(interval=1, metric=['bbox'])
evaluation = dict(interval=1, metric='bbox', save_best='auto') 
