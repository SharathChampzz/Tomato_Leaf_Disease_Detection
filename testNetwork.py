import os
import warnings
warnings.simplefilter("ignore")
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from flask import Flask, request, render_template
from keras import backend as K
K.clear_session()

##global result, percentage
ht = 28
wd = 28
classNames = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
              "Potato___healthy"]
totClass = len(classNames)

mdl = r"leafDiease.model"
img = r"M:\Python 3.7.4\Flask1\static\temp.JPG"

image = cv2.imread(img)
orig = image.copy()

# pre-process the image for classification
try:
    image = cv2.resize(image, (ht, wd))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
except Exception as e:
    print("Error Occured : ", e)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(mdl)
(zero, one, two, three) = model.predict(image)[0]
K.clear_session()
prob = [zero, one, two, three]
maxProb = max(prob)
maxIndex = prob.index(maxProb)
label = classNames[maxIndex]
proba = maxProb
for i in range(0, totClass):
    print(f'{classNames[i]} : {prob[i]}')

##label = "{}: {:.2f}%".format(label, proba * 100)
result = label
percentage = proba * 100
print(f'Result : {result}')
print(f'Accyracy : {percentage} %')
