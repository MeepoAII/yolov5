import os
import numpy as np
from numpy.core.shape_base import _block_format_index
import pandas as pd
from tqdm import tqdm

root_path = '/media/sata/public-data/data/kaggle/global_wheat_detection'
train_csv_path = os.path.join(root_path, 'train.csv')

df = pd.read_csv(train_csv_path)
bboxs = np.stack(df['bbox'].apply(lambda x : np.fromstring(x[1:-1], sep=','))) # excelent work!

# add a new column represent x, y, w, h
for i, column in enumerate(['x', 'y', 'w', 'h']):
    df[column] = bboxs[:,i]

df.drop(columns=['bbox'], inplace=False)

# add x_center, y_center, classes for yolov5 format

df['x_center'] = df['x'] + df['w']/2
df['y_center'] = df['y'] + df['h']/2

df['classes'] = 0

df = df[['image_id', 'x', 'y', 'w', 'h', 'x_center', 'y_center', 'classes']]

index = list(set(df['image_id']))

# all 3373 images


y5_format_path = '/media/sata/public-data/data/kaggle/gwd_yolov5'
# create labels


for name, mini in tqdm(df.groupby('image_id')):
    # print(name, mini)
    row = mini[['classes', 'x_center', 'y_center', 'w', 'h']].astype(float).values;
    row = row/1024
    row = row.astype(str)

    # may cause bug (class is int ???)
    # row[:, 0] = '0'     # set classes = 0
    txt_path = os.path.join(y5_format_path, 'labels', name+'.txt')
    with open(txt_path, 'w+') as f:
        for j in range(len(row)):
            text = ' '.join(row[j])
            f.write(text)
            f.write('\n')


