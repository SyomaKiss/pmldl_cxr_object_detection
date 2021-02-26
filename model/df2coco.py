import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def df2coco(csv_path, output_path=None, img_prefix='/home/semyon/data/VinBigData/train'):
    '''
    Transforms dataset to COCO format, drops 14th class.
    :param csv_path: input file with bboxes for each image.
    :param out_path: where to save resultant json
    :param img_prefix: where to read files to take image shape information
    :return:
    '''
    if output_path is None:
        output_path = csv_path + '.coco.json'

    df = pd.read_csv(csv_path)

    annotations = []
    images = []
    obj_count = 0

    CLASSES = ('Aortic enlargement',
         'Atelectasis',
         'Calcification',
         'Cardiomegaly',
         'Consolidation',
         'ILD',
         'Infiltration',
         'Lung Opacity',
         'Nodule/Mass',
         'Other lesion',
         'Pleural effusion',
         'Pleural thickening',
         'Pneumothorax',
         'Pulmonary fibrosis')
    class2idx = {v: k for k, v in dict(enumerate(CLASSES)).items()}
    try:
        class2idx = df.groupby('class_name').mean().class_id.to_dict()
        class2idx.pop('No finding')
    except:
        pass

    image_names = df.image_id.unique()

    img_prefix = Path(img_prefix)
    for idx, name in enumerate(tqdm(image_names)):
        filename = f'{name}.npy'
        filepath = img_prefix / filename

        height, width = np.load(filepath).shape

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for _, row in df[df.image_id == name].iterrows():
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[0, 0, 1, 1],
                area=1,
                iscrowd=0)
            if 'class_id' in row:
                if row.class_id != 14:
                    x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values.ravel().astype(np.uint64)
                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=row.class_id,
                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        area=(x_max - x_min) * (y_max - y_min),
                        iscrowd=0)

            annotations.append(data_anno)
            obj_count += 1
            
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': value, 'name': key} for key, value in class2idx.items()],
    )
    json.dump(coco_format_json, open(output_path, 'w'), cls=NpEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", help="path to initial .csv file", required=True)
    parser.add_argument("--output_path", help="path to directory to save .json in coco format")
    parser.add_argument("--img_prefix", help="path to directory with images", required=True)
    args = parser.parse_args()

    df2coco(args.input_path, args.output_path, img_prefix=args.img_prefix)
