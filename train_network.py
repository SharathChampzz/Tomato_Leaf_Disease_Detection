import matplotlib
matplotlib.use("Agg")
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.utils.np_utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time


## Expected Changes
resolution = 75 # Change resolution of image based on your interest
dataSet = r'M:\pythonD\PlantVillageFullDataBase\Tomato'  # Data set Location


EPOCHS = 25



# initialize the number of epochs to train for, initia learning rate,
# and batch size
INIT_LR = 1e-3
BS = 32


heightt = resolution
widthh = resolution


modelName = 'LeafDisease75x75.h5'
plotName = 'OutcomePlot.png'


### Collecting Class Names
pathh = dataSet  #dataSet
print(pathh)
cnames = os.listdir(pathh)
#Total Classes Automatically Takes 
totClass = len(cnames)
print(f'Total Classes = {totClass}')
print(f'Available Classes : {cnames}')

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataSet)))
random.seed(42)
random.shuffle(imagePaths)

imageFailed = 0
# loop over the input images
for imagePath in imagePaths:
        try:
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (heightt, widthh))
                image = img_to_array(image)
                lbl = imagePath.split(os.path.sep)[-2]
                label = cnames.index(lbl);
                data.append(image)
                labels.append(label)
        except Exception as e:
                imageFailed += 1
                print(f'Images Failed To Load  Count = {imageFailed}')
                print(e)
print('List OF data and labels are created')	
print(f'Data : {len(data)}')	
print(f'Label : {len(labels)}')
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print('[INFO] Spliting Data..')
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=totClass)
testY = to_categorical(testY, num_classes=totClass)

# construct the image generator for data augmentation
print('[INFO] Augmenting Data..')
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] Compiling Model...")
model = LeNet.build(width=widthh, height=heightt, depth=3, classes=totClass)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] Training Network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
t = time.time()
try:
    export_path = "/tmp/saved_models/{}".format(int(t))
    export_saved_model(model, export_path)
except Exception as e:
    print('Saving Alternatively...')
    model.save("LeafDisease.h5")
    print('Saved')
##model.save(modelName)
print('Saved..!')
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Leaf Disease Detection")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plotName)
