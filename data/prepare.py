# ==== Initialisation ====
import os
import itertools as it
import pandas as pd
import py.similarity as sm

# ==== Prepare image database using training samples ====
img_dir = 'data/train'
img_ids = os.listdir(img_dir)  # setup image name as image id
img_ids = list(it.compress(img_ids, [x.endswith(".jpg") for x in img_ids]))
img_pth = [os.path.join(img_dir, x) for x in img_ids]
db = pd.DataFrame(list(zip(img_ids, img_pth)), columns=['id', 'file'])

# feature extraction (1-2 sec/image) ====
db['feature'] = db.file.apply(sm.feature_extraction)
db.head()

# ==== save ====
db.to_pickle('data/db.pkl')
