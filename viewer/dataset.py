from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from app.helpers import expand_df_if_needed

# meta_path = "/home/semyon/data/VinBigData/train_meta.csv"
class CXRDatasetrFull(Dataset):
    def __init__(self, root, csv_path='train.csv', csv_meta_path='train_meta.csv', df=None):
        '''
        param: csv_path: 
        param: meta_path: initial resolutions
        '''
        self.root = Path(root)
        self.df = df if df is not None else pd.read_csv(csv_path)
        self.df = expand_df_if_needed(self.df)
        # self.df_meta = pd.read_csv(csv_meta_path)
        self.image_ids = self.df.image_id.unique()
        self.classes = sorted(self.df.class_name.unique())

    def __len__(self, ):
        return len(self.image_ids)

    def __getitem__(self, idx):
        name = self.image_ids[idx]
        tmp = self.df[self.df.image_id == name].copy()
        tmp = tmp[tmp.removed == 0].copy()
        # print()
        # print(tmp.to_markdown())

        path = self.root / f'{name}.npy'
        img = np.load(str(path))

        # rescale boxes

        tmp = tmp.sort_values(['class_name', 'rad_id'])

        meta = tmp[['class_name', 'class_id', 'rad_id', 'new_class']].to_dict('list')
        bboxes = tmp[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(np.uint16)

        return {'image': img, **meta, 'bboxes': bboxes, 'box_ids': list(tmp.index)}

    def get_idx_by_name(self, name):
        return list(self.image_ids).index(name)

    def get_name_by_idx(self, idx):
        return self.image_ids[idx]



if __name__ == '__main__':
    csv_path = '/home/semyon/data/VinBigData/train.csv'
    root = '/home/semyon/data/VinBigData/train/'
    d = CXRDatasetrFull(root, csv_path)
    entry = d[2]
    img = entry['image']
    rad_ids = entry['rad_id']
    class_names = entry['class_name']
    class_id = entry['class_id']
    bboxes = entry['bboxes']
    print(class_names)