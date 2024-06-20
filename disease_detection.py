# Dataset from: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data
import random
import os
import cv2
import numpy as np
import keras
import glob
from PIL import Image

num_images = 1500    #number of images from each class that will be used
SIZE = 64

dataset = []
label = []
labelCount = 0

'''adds a random sample of 1500 images from each
   class into the dataset list, resizing each image,
   and putting each in RGB mode
   
   corresponding labels (0-9) are also added into the label list'''
def load_images(dir_path):
    temp = []
    #print(temp)
    path = dir_path
    for file in glob.glob(path):
        image = cv2.imread(file)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        temp.append(np.array(image))
    random.seed(5)
    image_list = random.sample(temp, num_images)
    global dataset
    for i in range(len(image_list)):
        dataset.append(image_list[i])
    global labelCount
    for i in range(num_images):
        label.append(labelCount)
    labelCount += 1

load_images('Apple__Healthy/*')
load_images('Apple__Rotten/*')
load_images('Banana__Healthy/*')
load_images('Banana__Rotten/*')
load_images('Mango__Healthy/*')
load_images('Mango__Rotten/*')
load_images('Orange__Healthy/*')
load_images('Orange__Rotten/*')
load_images('Strawberry__Healthy/*')
load_images('Strawberry__Rotten/*')





