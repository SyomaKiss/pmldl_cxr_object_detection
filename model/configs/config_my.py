# The new config inherits a base config to highlight the necessary modification
# _base_ = '/home/semyon/projects/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_123epochs.py', '../_base_/default_runtime.py'
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=14),))

# Modify dataset related settings
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
        ann_file='/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-1234.json'),
    val=dict(
        img_prefix='/home/semyon/data/VinBigData/train',
        classes=classes,
        ann_file='/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-0.json'),
    test=dict(
        img_prefix='/home/semyon/data/VinBigData/train',
        classes=classes,
        ann_file='/home/semyon/data/VinBigData/coco_formats/weighted_boxes_fusion_iou-0.20_fold-0.json'),
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'