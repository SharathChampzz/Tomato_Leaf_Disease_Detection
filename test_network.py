# import the necessary packages
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
from os import listdir
import pickle
ht=75
wd=75
className = []

# Give the trained dataset path here it will just collect the labels 
# or directly create a list with class names
pathh = r"M:\pythonD\PlantVillageFullDataBase\Tomato"
classNames = listdir(pathh)
totClass = len(classNames)
print(classNames)
print(totClass)

mdl = r"M:\pythonD\15Days\7cls\LeafDisease.h5"

# Testing image path 
im = r"M:\pythonD\tomTargetSPot.jpg"

image = cv2.imread(im)
orig = image.copy()

# pre-process the image for classification
try:
    image = cv2.resize(image, (ht, wd))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
except Exception as e:
    print("Error Occured : ",e)
    


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(mdl)

# classify the input image
# How wany classes do you have ?  Create those many variables here. 
(zero, one,two, three,four,five,six,seven, eight,nine) = model.predict(image)[0]
prob = [zero, one,two, three,four,five,six,seven, eight,nine] # also add here & these are not string variables

maxProb = max(prob)
maxIndex = prob.index(maxProb)
label = classNames[maxIndex]
proba = maxProb
for i in range(0,totClass):
    print(f'{classNames[i]} : {prob[i]}')
    
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
