import copy
import numpy as np
import os
import os.path as osp
from datetime import datetime, timedelta

from . import BBOX_REMOVED_COL_NAME, NEW_CLASS_COL_NAME, DONE_COL_NAME


def create_log_dir(log_dir, csv_path, readonly=False):
    os.makedirs(log_dir, exist_ok=True)
    _, filename = osp.split(csv_path)

    time_str = (datetime.now() + timedelta(hours=3)).strftime("%d_%m_%Y-%H_%M_%S")
    new_dir_name = f"run_on-{filename}_at-{time_str}" + ('--READONLY' if readonly else '')
    current_log_dir = osp.join(log_dir, new_dir_name)

    os.makedirs(current_log_dir, exist_ok=True)

    new_csv_path = osp.join(current_log_dir, filename)
    return new_csv_path


def expand_df_if_needed(df):
    if DONE_COL_NAME not in df:
        df[DONE_COL_NAME] = df.image_id.apply(lambda x: 0)
    if NEW_CLASS_COL_NAME not in df:
        df[NEW_CLASS_COL_NAME] = df.class_name.apply(lambda x: x)
    if BBOX_REMOVED_COL_NAME not in df:
        df[BBOX_REMOVED_COL_NAME] = df.image_id.apply(lambda x: 0)
    return df


def get_from_dataset(dataset, idx):
    if isinstance(idx, int):
        entry = dataset[idx]
    else:
        entry = dataset.get_by_name(idx)

    img = entry["image"]
    rad_ids = entry["rad_id"]
    class_names = entry["class_name"]
    class_id = entry["class_id"]
    bboxes = entry["bboxes"]
    box_ids = entry["box_ids"]
    new_class_names = entry["new_class"]
    return img, rad_ids, class_names, class_id, bboxes, box_ids, new_class_names


def what_exactly_triggered(context, part_of_id):
    return (
        context.triggered
        and len(context.triggered) == 1
        and part_of_id in context.triggered[0]["prop_id"]
    )


def change_class(class_names, box_ids, box_id, new_class):
    """
    Change class name for particular bbox
    :param class_names:
    :param box_ids:
    :param box_id:
    :param new_class:
    :return:
    """
    inlist_idx = box_ids.index(box_id)
    class_names[inlist_idx] = new_class
    return class_names


def remove_boxes(box_ids_to_remove, box_ids, lists):
    """
    Remove one entry from all arguments after 'box_ids'
    :param box_id:
    :param box_ids:
    :param lists: list of lists of equal length
    :return:
    """
    ret = list(list(i) for i in lists)
    box_ids = copy.deepcopy(box_ids)
    for box_id in box_ids_to_remove:
        inlist_idx = box_ids.index(box_id)
        for i, _ in enumerate(lists):
            ret[i].pop(inlist_idx)
            box_ids.pop(inlist_idx)
    return list(np.array(i) for i in ret)
