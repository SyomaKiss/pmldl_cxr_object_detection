import numpy as np
import cv2

import pandas as pd

from tqdm.notebook import tqdm
from ensemble_boxes import weighted_boxes_fusion

def get_fused_boxes(image_id, records, conf_col_name=None, iou_thr = 0.2, skip_box_thr = 0, only_one=False):

    all_rad_ids = records.groupby('image_id')['rad_id'].agg(lambda x: ' '.join([str(i) for i in np.unique(x)])).iloc[0]

    if records.groupby('image_id').mean()['class_id'].values[0] == 14:
        tmp = records[['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max']].copy()
        tmp = tmp.iloc[:1]
        tmp['rad_id'] = all_rad_ids
        return tmp
    
    boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
    pix_multiplier = pd.DataFrame([records.width,records.height,records.width,records.height]).T
    boxes = [(boxes/(pix_multiplier)).values.tolist()]
    labels = [records["class_id"].tolist()]
    scores = [[1]*len(records)]
    if conf_col_name is not None:
        scores = [records[conf_col_name].tolist()]
    weights = [1]

    # If we demand only one of the label per image, we set iou threshold to 0
    if only_one:
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=0, skip_box_thr=skip_box_thr)
    else:
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes * pix_multiplier.iloc[:len(boxes),:]
    boxes.columns = ['x_min', 'y_min', 'x_max', 'y_max']
    boxes['class_id'] = labels.astype(int)
    boxes['image_id'] = image_id
    boxes['rad_id'] = all_rad_ids
    boxes['conf'] = scores
    if conf_col_name == 'rad_id':
        boxes['rad_id'] = scores
    return boxes

def ensemble_bboxes(input_path, output_path=None, conf_col_name='rad_id', iou_threshold=0.2, meta_path='/home/semyon/data/VinBigData/train_meta.csv', verbose=False):
    if isinstance(input_path, pd.DataFrame):
        df = input_path.copy()
    else:
        df = pd.read_csv(input_path)
        if output_path is None:
            output_path = input_path + '_bboxes_fusion_iou-{}.csv'.format(iou_threshold)
    
    
    meta_df = pd.read_csv(meta_path).set_index('image_id')
    df['height'] = df.image_id.apply(lambda x: meta_df.loc[x, 'rows'])
    df['width'] = df.image_id.apply(lambda x: meta_df.loc[x, 'columns'])
    
    class2id = df[['class_name', 'class_id']].groupby('class_name').mean().to_dict()['class_id']
    id2class = {v:k for k,v in class2id.items()}
    
    image_ids = df.image_id.unique()

    l = []
    for image_id in tqdm(image_ids):
        tmp = df[df.image_id == image_id].copy()
        l.append(get_fused_boxes(image_id, tmp, conf_col_name=conf_col_name, iou_thr = iou_threshold))


    new = pd.concat(l).reset_index(drop=True)
    new['class_name'] = new.class_id.apply(lambda x: id2class[x])
    new = new[['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min',
           'x_max', 'y_max', 'conf']]
    
    if output_path is not None:
        new.to_csv(output_path, index=False)
    
    return new