# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:22:24 2019

@author: deeplearning
"""
import os
import pandas as pd
import imgOps as img

# Prepare Database
imageFolder = 'images'
imageId = os.listdir(imageFolder)
imagePath = [os.path.join(imageFolder,x) for x  in os.listdir(imageFolder)]

data = pd.DataFrame(list(zip(imageId,imagePath)), columns=['imageId','imagePath'])

# feature extraction (This might take 1 to 2 sec per image or less :) )
data['feature'] = data.imagePath.apply(img.img2vec)

# Reverse Image search
queryPath = "images/cup1.jpg"
query = img.img2vec(queryPath)

data['match'] = data.feature.apply(lambda x: img.cos_sim(query,x))

data = data.sort_values(by=['match'], ascending=False).reset_index(drop=True)

results = data.head(6)
# Show Results
img.show_images([queryPath], ['Query Image'])
img.show_images(images=results['imagePath'], titles=results['match'].astype(str)+" %", cols=3)