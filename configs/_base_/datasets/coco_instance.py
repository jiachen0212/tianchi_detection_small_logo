# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/home/jiachen/data/tc_cc33/logo/'
classes = ('冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelme2coco/train.json',
        img_prefix=data_root + 'train/images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelme2coco/val.json',
        img_prefix=data_root + 'train/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotations/instances_val2017.json',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
