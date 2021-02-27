_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'
# dataset settings

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
img_scale = (320*2, 320*2)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
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

dataset_type = 'COCODataset'

classes = ('Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
           'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
           'Pulmonary fibrosis')

lr_config = dict(warmup_iters=2000)
optimizer = dict(lr=0.001)
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
data = dict(
    train=dict(
        img_prefix='/home/semyon/data/VinBigData/train',
        classes=classes,
        pipeline=train_pipeline,
        ann_file='/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-1234.json'),
    val=dict(
        img_prefix='/home/semyon/data/VinBigData/train',
        classes=classes,
        pipeline=test_pipeline,
        ann_file='/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-0.json'),
    test=dict(
        img_prefix='/home/semyon/data/VinBigData/train',
        classes=classes,
        pipeline=test_pipeline,
        ann_file='/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-0.json'),
)