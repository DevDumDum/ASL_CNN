import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import matplotlib.pyplot as plt


batch_size = 32
img_height = 100
img_width = 100
model_name = "asl_recog_E15-100x100"
cwd = os.getcwd()
dir_ds = cwd+'/'+model_name+'.model'

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = tf.keras.models.load_model(dir_ds)
def predict_img(img):
    img = cv2.resize(img, (img_height,img_width), interpolation= cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = tf.expand_dims(img, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions)

    letter = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    temp_img = img
    return letter, confidence, img, temp_img