import urllib.request, io, os
import pandas as pd, numpy as np
from PIL import Image

url = 'https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main/bottle/test/good/000.png'
data = urllib.request.urlopen(url).read()
gt_img = Image.open(io.BytesIO(data)).convert('RGB').resize((128, 128))
gt_arr = np.array(gt_img).astype(float) / 255.0

test_df = pd.read_csv('test.csv')
bottle_test = test_df[test_df.category == 'bottle']

best_mse = 999999
best_row = None
for idx, row in bottle_test.iterrows():
    p = os.path.join('clips', row['path'])
    im = Image.open(p).convert('RGB').resize((128, 128))
    arr = np.array(im).astype(float) / 255.0
    mse = float(np.mean((gt_arr - arr)**2))
    if mse < best_mse:
        best_mse = mse
        best_row = row

print("Best match for bottle/test/good/000.png:")
print("ID:", best_row['id'], "Path:", best_row['path'], "MSE:", best_mse)
