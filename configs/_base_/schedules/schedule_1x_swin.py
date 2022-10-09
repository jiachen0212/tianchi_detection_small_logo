optimizer = dict(type='Adm', lr=0.03)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[40, 60, 70])
runner = dict(type='EpochBasedRunner', max_epochs=80)
