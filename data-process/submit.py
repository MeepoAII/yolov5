import pandas as pd
import os

label_path = '/home/meepo/myfiles/code/meepo/yolov5/runs/detect/exp3/labels'

df = pd.DataFrame(columns=['image_id', 'PredictionString'])

for file in os.listdir(label_path):
    full_path = os.path.join(label_path, file)
    print(full_path)
    image_id = file.split('.')[0]
    print(image_id)
    break

# df.to_csv('submission.csv', header=True, index=False)
