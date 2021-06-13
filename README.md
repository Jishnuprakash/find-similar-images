# Find similar images

## About
The idea here is to search similar images from a image database by uploading a target picture or taking a photo. Additional attributes (if present) can also be retrieved. 

## Technique 
Features extracted from image is used to find similarities. I have used Pytorch `resnet18` (pre-trained model) for this exercise. 

Similarity is calculated between the target and reference images using `cosine similarity`.

## Setup environment 
Use "help/setup-environment.txt" (`platform: win-64`) in the following command to setup your python environment: 
`$ conda create --name <env> --file <this file>` 

## Result
- Query Image
![](https://github.com/Jishnuprakash/find-similar-images/blob/main/help/test_image.png)

- Results
![](https://github.com/Jishnuprakash/find-similar-images/blob/main/help/results.png)

## Reference
Please run `example.py`