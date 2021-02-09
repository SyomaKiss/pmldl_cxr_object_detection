import pandas as pd
import cv2
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, DataLoader


class CXRDatasetr512(Dataset):
    def __init__(self, root, csv_path='train.csv'):
        '''
        param: csv_path: 
        param: meta_path: initial resolutions
        '''
        self.root = Path(root)
        self.df = pd.read_csv(csv_path)
        self.df_meta = pd.read_csv(self.root / 'train_meta.csv')
        self.ids = self.df.image_id.unique()

    def __len__(self, ):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        tmp = self.df[self.df.image_id == name].copy()
        path = self.root / 'train' / f'{name}.png'
        img = cv2.imread(str(path))

        # rescale boxes
        row = self.df_meta[self.df_meta.image_id == name].iloc[0]
        orig_shape = (row.dim0, row.dim1)
        tmp.loc[:, 'x_min'] = tmp.x_min / orig_shape[1] * img.shape[1]
        tmp.loc[:, 'x_max'] = tmp.x_max / orig_shape[1] * img.shape[1]
        tmp.loc[:, 'y_min'] = tmp.y_min / orig_shape[0] * img.shape[0]
        tmp.loc[:, 'y_max'] = tmp.y_max / orig_shape[0] * img.shape[0]

        tmp = tmp.sort_values(['class_name', 'rad_id'])

        meta = tmp[['class_name', 'class_id', 'rad_id']].to_dict('list')
        bboxes = tmp[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(np.uint16)

        return {'image': img, **meta, 'bboxes': bboxes}

    def get_by_name(self, name):
        return self.__getitem__(list(self.ids).index(name))




if __name__ == '__main__':
    csv_path = '/Users/semenkiselev/Documents/dev/job/kaggle/train.csv'
    root = '/Users/semenkiselev/Documents/dev/job/kaggle/archive/'
    d = CXRDatasetr512(root, csv_path)
    entry = d[3]
    img = entry['image']
    rad_ids = entry['rad_id']
    class_names = entry['class_name']
    class_id = entry['class_id']
    bboxes = entry['bboxes']
    print(class_names)