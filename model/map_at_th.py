import pandas as pd
import numpy as np
import os.path as osp
from tqdm.notebook import tqdm

from map_boxes import mean_average_precision_for_boxes

def norm_coordinates(df, meta_df):
    '''
    INPLACE !!!
    '''
    if max([df.x_min.max(), df.x_max.max(), df.y_min.max(), df.y_max.max()]) <= 1:
        raise ValueError('Bbox coordinates are already normalized')
        return
    df['height'] = df.image_id.apply(lambda x: meta_df.loc[x, 'rows'])
    df['width'] = df.image_id.apply(lambda x: meta_df.loc[x, 'columns'])
    df['x_min'] = df.x_min / df.width
    df['x_max'] = df.x_max / df.width
    df['y_min'] = df.y_min / df.height
    df['y_max'] = df.y_max / df.height


def get_mean_average_precision(annotation_path, predictions_path, iou_threshold=0.4, meta_path='/home/semyon/data/VinBigData/train_meta.csv', verbose=False):
    '''
    param: annotation_path: path to .csv with columns ['image_id', 'class_name', 'x_min', 'x_max', 'y_min', 'y_max']
    param: predictions_path: path to .csv with columns ['image_id', 'class_name', 'rad_id', 'x_min', 'x_max', 'y_min', 'y_max'], where 'rad_id' contains confidence
    '''
    if isinstance(annotation_path, pd.DataFrame) and isinstance(predictions_path, pd.DataFrame):
        ann_df = annotation_path.copy()
        pred_df = predictions_path.copy()
    else:
        ann_df = pd.read_csv(annotation_path)
        pred_df = pd.read_csv(predictions_path)

    meta_df = pd.read_csv(meta_path).set_index('image_id')
    
    # inplace norm coordinates
    norm_coordinates(ann_df, meta_df)
    norm_coordinates(pred_df, meta_df)
    
    # annotations
    new_cols = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']
    old_cols = ['image_id', 'class_name', 'x_min', 'x_max', 'y_min', 'y_max']
    for new_col_name, old_col_name in zip(new_cols, old_cols):
        ann_df[new_col_name] = ann_df[old_col_name]
    # predictions
    new_cols = ['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']
    old_cols = ['image_id', 'class_name', 'rad_id', 'x_min', 'x_max', 'y_min', 'y_max']
    for new_col_name, old_col_name in zip(new_cols, old_cols):
        pred_df[new_col_name] = pred_df[old_col_name]
        
    ann = ann_df[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
    pred = pred_df[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, pred, iou_threshold=iou_threshold, verbose=verbose)
    return mean_ap, average_precisions