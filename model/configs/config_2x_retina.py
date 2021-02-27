# The new config inherits a base config to highlight the necessary modification
# _base_ = '/home/semyon/projects/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained="torchvision://resnet101",
    backbone=dict(depth=101),
    bbox_head=dict(num_classes=14),
)
# SCHEDULER
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24


optimizer = dict(type='Adam', lr=0.0002)


# DATASET
dataset_type = "COCODataset"

classes = (
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
)

# lr_config = dict(warmup_iters=2000)
# optimizer = dict(lr=0.00)
log_config = dict(
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
data = dict(
    train=dict(
        img_prefix="/home/semyon/data/VinBigData/train",
        classes=classes,
        ann_file="/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-1234.json",
    ),
    val=dict(
        img_prefix="/home/semyon/data/VinBigData/train",
        classes=classes,
        ann_file="/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-0.json",
    ),
    test=dict(
        img_prefix="/home/semyon/data/VinBigData/train",
        classes=classes,
        ann_file="/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-0.json",
    ),
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
img_size = (512,512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
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