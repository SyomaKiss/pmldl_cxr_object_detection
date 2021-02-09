import os
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import pydicom
import numpy as np
import pandas as pd
from pydicom.pixel_data_handlers.util import apply_voi_lut



def read_xray(image_id, data_dir, voi_lut=True, fix_monochrome=True):
    
    path = data_dir / f'{image_id}.dicom'
    if not path.exists():
        return 'n'
    dicom = pydicom.read_file(path, 
                                  stop_before_pixels=True
                                 )
    
    if not (data_dir / f'{image_id}.npy').exists():
        dicom = pydicom.read_file(path)

        # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data.astype(np.uint16)
        np.save(data_dir / f'{image_id}.npy', data)
        
        
    
    
    meta = dict(
        image_id=image_id,
        rows=dicom.Rows,
        columns=dicom.Columns,

        sex=dicom.get('PatientSex'),
        age=dicom.get('PatientAge'),
    )
    for label in ['WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope']:
        meta[label] = dicom.get(label)
        
    # Uncomment to remove dicom
#     os.remove(path)
    return 'a'

def meta2df(meta):
    def correct_age(age_str):
        if age_str is None:
            return -1

        x = age_str.replace('Y', '').replace('D', '')
        if x == '':
            x = -1
        x = int(x)
        if x == 0:
            x = -1
        return x
    
    data = {key: [m[key] for m in meta] for key in meta[0]}
    data['age'] = list(map(correct_age, data['age']))
    
#     image_id = [m['image_id'] for m in meta]
#     rows = [m['rows'] for m in meta]
#     columns = [m['columns'] for m in meta]
#     sex = [m['sex'] for m in meta]
#     age = [correct_age(m['age']) for m in meta]
    
#     df = pd.DataFrame({
#         'image_id': image_id,
#         'rows': rows,
#         'columns': columns,
#         'sex': sex,
#         'age': age,
#     })
    df = pd.DataFrame(data)
    return df


def main():
    root = Path('/home/semyon/data/VinBigData')
    
    df = pd.read_csv(root / 'train.csv')
    image_ids = df['image_id'].unique()
    read_xray_fn = partial(read_xray, data_dir=root / 'train')
    with Pool(32) as p:
        meta = p.map(read_xray_fn, image_ids)
#     df_train = meta2df(meta)
#     df_train.to_csv(root / 'train_meta.csv', index=False)
    print('train done')
    test_dir = root / 'test'
    image_ids = [p.stem for p in test_dir.glob('*')]
    read_xray_fn = partial(read_xray, data_dir=test_dir)
    with Pool(32) as p:
        meta = p.map(read_xray_fn, image_ids)
#     df_train = meta2df(meta)
#     df_train.to_csv(root / 'test_meta.csv', index=False)


if __name__ == '__main__':
    main()
