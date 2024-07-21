# Dataset from: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data
import random
import os
import cv2
import numpy as np
import keras
import glob
from PIL import Image
from keras import Sequential
from keras.src.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

num_images = 1500    #number of images from each class that will be used
SIZE = 32   #resize images to this size

dataset = []   #list of images
label = []     #list of labels
labelCount = 0   #0-9, for 10 classes, 1st class will have label 0

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


#load fruit images
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




def random_forest_model():
    # flatten dataset
    temp = np.array(dataset)
    flattened_dataset = temp.reshape(len(dataset), -1)

    # split data (80% training, 20% testing)
    x_train, x_test, y_train, y_test = (train_test_split(flattened_dataset, label,
                                                         train_size=0.8, random_state=42))

    # create classifier using default params
    classifier = RandomForestClassifier(n_estimators=100)

    # fit model to training data
    classifier.fit(x_train, y_train)
    # predict
    y_pred = classifier.predict(x_test)

    # model accuracy
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))

def cnn_model():
    # np arrays
    dataset_np = np.array(dataset)
    label_np = np.array(label)

    # normalize
    dataset_np = dataset_np / 255.0

    # categorical
    label_np = to_categorical(label_np, num_classes=10)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(dataset_np, label_np, test_size=0.2, random_state=42)

    # creating a sequential model
    model = Sequential()

    # convolutional layers and having ReLU activation
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # flattening 3D output to 1D output
    model.add(Flatten())

    # adding additional layers with ReLU and softmax activation
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # compiling the model with Adam optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary)

    # callbacks for checking the model
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    mc = keras.callbacks.ModelCheckpoint('best_model.keras',
                                         mode='max', verbose=1, save_best_only=True)

    # training the model
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[es, mc])

    # evaluating the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("CNN Accuracy: %.2f%%" % (scores[1] * 100))
    print("CNN Loss: %.2f" % scores[0])

# run models
svm_model()
random_forest_model()





