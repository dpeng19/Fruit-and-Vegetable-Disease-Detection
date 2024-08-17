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
from skimage import io

num_images = 1500  # number of images from each class that will be used
SIZE = 32  # resize images to this size

dataset = []  # list of images
label = []  # list of labels
labelCount = 0  # 0-9, for 10 classes, 1st class will have label 0

'''adds a random sample of 1500 images from each
   class into the dataset list, resizing each image,
   and putting each in RGB mode

   corresponding labels (0-9) are also added into the label list'''


def load_images(dir_path):
    temp = []
    # print(temp)
    folder = dir_path + '/'
    images = os.listdir(folder)
    for i, image_name in enumerate(images):
        image = io.imread(folder + image_name)
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


# load fruit images
load_images('Apple__Healthy')
load_images('Apple__Rotten')
load_images('Banana__Healthy')
load_images('Banana__Rotten')
load_images('Mango__Healthy')
load_images('Mango__Rotten')
load_images('Orange__Healthy')
load_images('Orange__Rotten')
load_images('Strawberry__Healthy')
load_images('Strawberry__Rotten')


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


svm_model()
random_forest_model()

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
                                         mode='min', verbose=1, save_best_only=True)

    # training the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[es, mc])

    # evaluating the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("CNN Accuracy: %.2f%%" % (scores[1] * 100))
    print("CNN Loss: %.2f" % scores[0])


# run models
svm_model()
random_forest_model()
cnn_model()


'''Below is code for data augmentation for classes with fewer images and for adding and applying'''
'''these images to train a cnn model. We train and test a cnn model to classify between 16'''
'''classes instead of 10.'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# image augmentation info
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'

)

'''This method augments images for a specified directory, saving those images in a '''
'''new directory with the suffix Augmented'''

def data_augment(dir):
    SIZE = 128
    x = []
    folder = dir + '/'
    images = os.listdir(folder)
    for i, image_name in enumerate(images):
        image = io.imread(folder + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        x.append(np.array(image))
    x = np.array(x)

    image_count = 0
    for batch in datagen.flow(x,
                              batch_size=16,
                              save_to_dir=dir + '_' + 'Augmented/',
                              save_prefix='aug',
                              save_format='png'
                              ):
        image_count = image_count + 1
        if image_count > 40:
            break


# augment images for each of these directories
data_augment('Bellpepper__Healthy')
data_augment('Bellpepper__Rotten')
data_augment('Carrot__Healthy')
data_augment('Carrot__Rotten')
data_augment('Cucumber__Healthy')
data_augment('Cucumber__Rotten')

dataset_augment_train = []
label_augment_train = []
dataset_augment_test = []
label_augment_test = []
NUM_TEST_IMAGES = 280  # will select 280 images for testing for the classes that have augmented iamges

'''This method will load the images for the classes with augmented images '''
'''into the train and test lists, both for the images and the labels'''
'''This also ensures that only real images are tested on and not augmented ones.'''


def load_images_augment(dir):
    temp = []

    folder = dir + '/'
    images = os.listdir(folder)
    for i, image_name in enumerate(images):
        image = io.imread(folder + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        temp.append(np.array(image))
    random.seed(5)
    # select 280 for testing, the rest for training
    train_image_list, test_image_list = train_test_split(temp, test_size=NUM_TEST_IMAGES, random_state=1)

    temp = []
    augmented_dir = dir + '_' + 'Augmented/'
    images = os.listdir(augmented_dir)
    for i, image_name in enumerate(images):
        image = io.imread(augmented_dir + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        temp.append(np.array(image))
    # select augmented images and add them in the training list, in addition to the real images,
    # total of 1000 images for training and testing
    augment_train = random.sample(temp, 1000 - len(test_image_list) - len(train_image_list))
    train_image_list = train_image_list + augment_train

    global labelCount
    global dataset_augment_train
    global dataset_augment_test
    global label_augment_test
    global label_augment_train
    for i in range(len(train_image_list)):
        dataset_augment_train.append(train_image_list[i])
        label_augment_train.append(labelCount)
    for i in range(len(test_image_list)):
        dataset_augment_test.append(test_image_list[i])
        label_augment_test.append(labelCount)
    labelCount += 1


# load images for the augmented datasets
load_images_augment('Bellpepper__Healthy')
load_images_augment('Bellpepper__Rotten')
load_images_augment('Carrot__Healthy')
load_images_augment('Carrot__Rotten')
load_images_augment('Cucumber__Healthy')
load_images_augment('Cucumber__Rotten')


def cnn_with_augmentation():


    # split data
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.28, random_state=0, stratify=label)

    for i in range(len(dataset_augment_train)):
        X_train.append(dataset_augment_train[i])
    for i in range(len(label_augment_train)):
        y_train.append(label_augment_train[i])
    X_train = np.array(X_train)
    X_train = X_train / 255.0
    y_train = np.array(y_train)

    y_train = to_categorical(y_train, num_classes=16)
    # append augment images for testing
    for i in range(len(dataset_augment_test)):
        X_test.append(dataset_augment_test[i])
    for i in range(len(label_augment_test)):
        y_test.append(label_augment_test[i])
    X_test = np.array(X_test)
    X_test = X_test / 255.0
    y_test = np.array(y_test)

    y_test = to_categorical(y_test, num_classes=16)
    # split into validation and testing from the testing data
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=4200, random_state=0, stratify=y_test)
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
    model.add(Dense(16, activation='softmax'))

    # compiling the model with Adam optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary)

    # callbacks for checking the model
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    mc = keras.callbacks.ModelCheckpoint('best_model.keras',
                                         mode='min', verbose=1, save_best_only=True)

    # training the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[es, mc])

    # evaluating the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Accuracy: %.2f%%" % (scores[1] * 100))
    print("CNN Loss: %.2f" % scores[0])

#run cnn for 16 classes classification
cnn_with_augmentation()