optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001, eps=1e-3)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[25, 40, 48])
runner = dict(type='EpochBasedRunner', max_epochs=50)