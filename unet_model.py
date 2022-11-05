# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:56:45 2022

@author: binda
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import random
import cv2
import tensorflow as tf
# from segmentation import build_unet, vgg16_unet, vgg19_unet, resnet50_unet, inception_resnetv2_unet, densenet121_unet
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score
from tensorflow import keras


x_train_folder = 'D:\\RID-master\\RID-master\\venv\\data\\csv_datasets_v3\\dataset\\train_images'

train_folder = 'D:\\RID-master\\RID-master\\venv\\data\\csv_datasets_v3\\dataset\\train_mask'
x_test_folder = 'D:\\RID-master\\RID-master\\venv\\data\\csv_datasets_v3\\dataset\\test_images'
test_folder = 'D:\\RID-master\\RID-master\\venv\\data\\csv_datasets_v3\\dataset\\test_mask'
print(x_train_folder, train_folder, x_test_folder, test_folder)
dim = 256

def input_target_split(x_train_folder, train_folder, dim):
    output = []
    output1 = []
    dataset = []
    files_roof = [geotif[:-4] for geotif in os.listdir(x_train_folder) if geotif[-4:] == '.png']
    files_roof1 = [geotif[:-4] for geotif in os.listdir(x_test_folder) if geotif[-4:] == '.png']
    files_png = [png[:-4] for png in os.listdir(x_train_folder) if png[-4:] == '.png']
    #print(files_png)
    missing_pngs_list = [geotif for geotif in files_roof if geotif in files_png]
    #print(missing_pngs_list)
    missing_pngs_list1 = [geotif for geotif in files_roof1 if geotif in files_png]

#print(len(a))
    

    for i, img in enumerate(missing_pngs_list):
        roof1 = os.path.join(x_train_folder, img + '.png')
        roof2 = os.path.join(train_folder, img + '.png')
        train_image = load_img(roof1, target_size=(dim,dim))
        train_img = img_to_array(train_image)
        train_img = train_img/255.0
        
        train_mask = load_img(roof2, target_size = (dim, dim), color_mode= 'grayscale')
        train_msk = img_to_array(train_mask)
        train_msk = train_msk/255.0
        #print(train_msk)
        dataset.append((train_img, train_msk))
        #print(dataset)
    X, Y = zip(*dataset)
    del train_image, roof1, roof2, train_mask, dataset
    return np.array(X), np.array(Y)


X, Y = input_target_split(x_train_folder, train_folder, dim)


A, B = input_target_split(x_test_folder, test_folder, dim)

print("Image Dimensions: ",X.shape)
print("Mask Dimensions: ",Y.shape)


plt.figure(figsize = (15 , 9))
n = 0
for i in range(15):
    n+=1
    plt.subplot(2 , 2, n)
    plt.subplots_adjust(hspace = 0.2 , wspace = 0.2)
    plt.imshow(X[i])
    plt.title('Image')
    
plt.figure(figsize = (15 , 9))
n = 0
for i in range(15):
    n+=1
    plt.subplot(2 , 2, n)
    plt.subplots_adjust(hspace = 0.2 , wspace = 0.2)
    plt.imshow(Y[i])
    plt.title('Masks')

datagen = ImageDataGenerator()
testgen = ImageDataGenerator()


X_train = X
Y_train = Y
X_test = A
Y_test = B
del X, Y, A, B
datagen.fit(X_train)
testgen.fit(X_test)

print(X_train.max(),
X_train.min(),
Y_test.max(),
Y_test.min(),
Y_train.max(),
Y_train.min(),
X_test.max(),
X_test.min())
#






def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    metric_value = tf.compat.v1.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value


input_shape = (dim, dim, 3)
model = build_unet(input_shape)
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001), loss=['binary_crossentropy'] , metrics=['accuracy'])

model_path = "D:\\RID-master\\RID-master\\venv\\data\\csv_datasets_v3\\dataset\\train_images\\unet.h5"
checkpoint = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 9,
                          verbose = 1,
                          restore_best_weights = True)




hist = model.fit(datagen.flow(X_train,Y_train,batch_size=16),
                                        validation_data=testgen.flow(X_test,Y_test,batch_size=24),
                                        epochs=10, callbacks=[earlystop, checkpoint])


plt.figure(figsize = (15 , 9))
n = 0
for i in range(0,15):
    n+=1
    plt.subplot(5 , 5, n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.3)
    plt.imshow(Y_test[i])
    plt.title('Masks')


fig= plt.figure(figsize = (15 , 9))
result = model.predict(X_test)
output = result[9]
output[output >= 0.5] = 1
output[output < 0.5] = 0
#print(output)


plt.subplot(1, 3, 1)
plt.imshow(X_test[9])

plt.subplot(1, 3, 2)
plt.imshow(Y_test[9])

plt.subplot(1, 3, 3)
plt.imshow(output)
fig.savefig('Image_output')



plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('plot')





