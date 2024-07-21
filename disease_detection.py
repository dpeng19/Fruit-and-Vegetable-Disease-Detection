"""
Support Vector Machines Model
Dataset from: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data
"""

import random
import cv2
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


NUM_IMAGES = 1500  # number of images from each class that will be used
SIZE = 32

dataset = []
label = []
labelCount = 0

'''
adds a random sample of 1500 images from each
   class into the dataset list, resizing each image,
   and putting each in RGB mode
   
   corresponding labels (0-9) are also added into the labels list'''


def load_images(dir_path):
    dir_path = "Fruit And Vegetable Diseases Dataset/" + dir_path
    temp = []
    # print(temp)
    path = dir_path
    for file in glob.glob(path):
        image = cv2.imread(file)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        temp.append(np.array(image))
    random.seed(5)
    image_list = random.sample(temp, NUM_IMAGES)
    global dataset
    for image in image_list:
        dataset.append(image)
    global labelCount
    for i in range(NUM_IMAGES):
        label.append(labelCount)
    labelCount += 1


# load in all images
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


###################################################

# method that runs SVC model
def svm_model():
    # flatten dataset
    temp = np.array(dataset)
    flattened_dataset = temp.reshape(len(dataset), -1)

    # split data (80% training, 20% testing)
    x_train, x_test, y_train, y_test = (train_test_split(flattened_dataset, label,
                                                         train_size=0.8, random_state=42))

    # create classifier using default params
    classifier = SVC()
    # fit model to training data
    classifier.fit(x_train, y_train)
    # predict
    y_pred = classifier.predict(x_test)

    # model accuracy
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))


# run model
svm_model()
