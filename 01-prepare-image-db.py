# Imports
import os
import pandas as pd
import py.img_operations as img

# Prepare Database
img_dir = 'data/train'
img_ids = os.listdir(img_dir)
img_pth = [os.path.join(img_dir,x) for x in img_ids]
df = pd.DataFrame(list(zip(img_ids,img_pth)), columns=['imageId','imagePath'])

# feature extraction (This might take 1 to 2 sec per image or less :) )
df['feature'] = df.imagePath.apply(img.img2vec)

# save
df.to_pickle('data/img_db.pkl')