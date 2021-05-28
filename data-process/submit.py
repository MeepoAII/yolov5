from numpy.core.numeric import full
import pandas as pd
import os

label_path = '/home/meepo/myfiles/code/meepo/yolov5/runs/detect/exp12/labels'

df = pd.DataFrame(columns=['image_id', 'PredictionString'])

for file in os.listdir(label_path):
    full_path = os.path.join(label_path, file)
    image_id = file.split('.')[0]
    prediction_string = ''
    with open(full_path, 'r') as f:
        for line in f:
            line = line.replace('\n', ' ')
            print(line)
            prediction_string += line
    df = df.append({'image_id':image_id, 'PredictionString':prediction_string}, ignore_index=True)
        

print(df)

# df.to_csv('submission.csv', header=True, index=False)
