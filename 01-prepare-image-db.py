#Imports
import os
import pandas as pd
import imgOps as img

#### Prepare Database ####
imageFolder = 'images'
imageId = os.listdir(imageFolder)
imagePath = [os.path.join(imageFolder,x) for x  in os.listdir(imageFolder)]

data = pd.DataFrame(list(zip(imageId,imagePath)), columns=['imageId','imagePath'])

# feature extraction (This might take 1 to 2 sec per image or less :) )
data['feature'] = data.imagePath.apply(img.img2vec)