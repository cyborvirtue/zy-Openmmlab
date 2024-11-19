default_scope = 'mmocr'

_base_ = [
    '../_base_/datasets/mlt2019_dataset.py',  # 导入 MLT2019 数据集配置
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_panet_resnet50_fpem-ffm.py',
]

# 检查点保存的配置
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20))

# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='ShortScaleAspectJitter', short_size=800, scale_divisor=32),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', max_angle=10),
    dict(type='TextDetRandomCrop', target_size=(800, 800)),
    dict(type='Pad', size=(800, 800)),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 测试数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='ShortScaleAspectJitter',
        short_size=800,
        scale_divisor=1,
        ratio_range=(1.0, 1.0),
        aspect_ratio_range=(1.0, 1.0)),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 导入 MLT2019 数据集配置
mlt2019_textdet_train = _base_.mlt2019_textdet_train
mlt2019_textdet_test = _base_.mlt2019_textdet_test

# 设置流水线
mlt2019_textdet_train.pipeline = train_pipeline
mlt2019_textdet_test.pipeline = test_pipeline

# 训练数据加载器
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=mlt2019_textdet_train)

# 验证和测试数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=mlt2019_textdet_test)

test_dataloader = val_dataloader

# 验证和测试评估器
val_evaluator = dict(
    type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=1, step=0.05))
test_evaluator = val_evaluator

# 自动调整学习率
auto_scale_lr = dict(base_batch_size=64)
