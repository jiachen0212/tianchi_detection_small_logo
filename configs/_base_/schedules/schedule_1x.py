optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[25, 40, 48])
runner = dict(type='EpochBasedRunner', max_epochs=50)


# # sgd
# # optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# # adam
# optimizer = dict(type='Adam', lr=0.02, weight_decay=0.0001)

# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',  # 'consine cyclic'
#     warmup='linear',
#     warmup_iters=2000,
#     warmup_ratio=0.001,
#     step=[30, 42, 48])

# # 余弦退火算法 有bug..
# # lr_config = dict(
# #     policy='CosineAnealing',
# #     warmup='exp',
# #     warmup_iters=1000,
# #     warmup_ratio=1.0 / 3,
# #     min_lr=0.0005)

# runner = dict(type='EpochBasedRunner', max_epochs=50)


# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#  policy='step',
#  warmup='constant', # 不再是linear
#  warmup_iters=800,
#  warmup_ratio=1.0 / 3, # 1/3倍的初始学习比率
#  step=[30, 42, 48])
# runner = dict(type='EpochBasedRunner', max_epochs=50)


# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,  # 这个貌似是cascade0.47的结果.
#     step=[25, 40, 48])

# runner = dict(type='EpochBasedRunner', max_epochs=50)

