# sgd
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# adam
optimizer = dict(type='Adam', lr=0.02, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[20, 35, 38])

# 余弦退火算法 有bug..
# lr_config = dict(
#     policy='CosineAnealing',
#     warmup='exp',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 3,
#     min_lr=0.0005)

runner = dict(type='EpochBasedRunner', max_epochs=40)