# Imports
import os
import pandas as pd
import py.img_operations as img

# Load image database
df = pd.read_pickle('data/img_db.pkl')
df.head()

# Load test image
test_image = "data/test/test-01.jpg"
feature = img.img2vec(test_image)
feature.shape

# Find cosine similarity of the test image with the whole image db
df['match'] = df.feature.apply(lambda x: img.cos_sim(feature,x))
df.describe()
df = df.sort_values(by=['match'], ascending=False).reset_index(drop=True) # Sort values
top = df.head(6) # Select top results

# Show Results
img.show_images([test_image], ['test Image'])
img.show_images(images=top['imagePath'], titles=top['match'].astype(str)+" %", cols=3)