import os.path as osp

import pandas as pd
import cv2
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from app.helpers import expand_df_if_needed


class CXRDatasetr512(Dataset):
    def __init__(
        self,
        root,
        csv_path="train.csv",
        df=None,
        meta_csv_path="/home/semyon/data/vinbigdata/archive/train_meta.csv",
    ):
        """
        param: csv_path:
        param: meta_path: initial resolutions
        """
        self.root = Path(root)
        self.df = df if df is not None else pd.read_csv(csv_path)
        self.df = expand_df_if_needed(self.df)
        dataset_dir, phase = osp.split(str(root))
        self.df_meta = pd.read_csv(osp.join(dataset_dir, f'{phase}_meta.csv'))
        self.image_ids = self.df.image_id.unique()

    def __len__(
        self,
    ):
        return len(self.image_ids)

    def __getitem__(self, idx):
        name = self.image_ids[idx]
        tmp = self.df[self.df.image_id == name].copy()
        tmp = tmp[tmp.removed == 0].copy()
        # print()
        # print(tmp.to_markdown())
        path = self.root / f"{name}.png"
        img = cv2.imread(str(path))

        # rescale boxes
        row = self.df_meta[self.df_meta.image_id == name].iloc[0]
        orig_shape = (row.dim0, row.dim1)
        tmp.loc[:, "x_min"] = tmp.x_min / orig_shape[1] * img.shape[1]
        tmp.loc[:, "x_max"] = tmp.x_max / orig_shape[1] * img.shape[1]
        tmp.loc[:, "y_min"] = tmp.y_min / orig_shape[0] * img.shape[0]
        tmp.loc[:, "y_max"] = tmp.y_max / orig_shape[0] * img.shape[0]

        tmp = tmp.sort_values(["class_name", "rad_id"])

        meta = tmp[["class_name", "class_id", "rad_id", "new_class"]].to_dict("list")
        bboxes = tmp[["x_min", "y_min", "x_max", "y_max"]].values.astype(np.uint16)

        return {"image": img, **meta, "bboxes": bboxes, "box_ids": list(tmp.index)}

    def get_idx_by_name(self, name):
        return list(self.image_ids).index(name)

    def get_name_by_idx(self, idx):
        return self.image_ids[idx]


if __name__ == "__main__":
    csv_path = "/Users/semenkiselev/Documents/dev/job/kaggle/train.csv"
    root = "/Users/semenkiselev/Documents/dev/job/kaggle/archive/"
    d = CXRDatasetr512(root, csv_path)
    entry = d[3]
    img = entry["image"]
    rad_ids = entry["rad_id"]
    class_names = entry["class_name"]
    class_id = entry["class_id"]
    bboxes = entry["bboxes"]
    print(class_names)
