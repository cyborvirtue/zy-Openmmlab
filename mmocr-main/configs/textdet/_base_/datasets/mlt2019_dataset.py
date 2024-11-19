# mlt2019_dataset.py

mlt2019_textdet_data_root = 'data/mlt2019'

# 训练集配置
mlt2019_textdet_train = dict(
    type='OCRDataset',
    data_root=mlt2019_textdet_data_root,
    ann_file='textdet_train.json',  # 训练集标注文件
    filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 过滤掉没有标注的图像和小于指定大小的标注
    pipeline=None  # 训练数据集的处理流水线将在模型配置中定义
)

# 测试集配置
mlt2019_textdet_test = dict(
    type='OCRDataset',
    data_root=mlt2019_textdet_data_root,
    ann_file='textdet_test.json',  # 测试集标注文件
    test_mode=True,  # 表示这是一个测试集
    pipeline=None  # 测试数据集的处理流水线将在模型配置中定义
)
