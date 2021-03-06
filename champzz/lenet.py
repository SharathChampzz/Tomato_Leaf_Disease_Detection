# import the necessary packages
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D , MaxPooling2D
##from keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation , Flatten , Dense
##from keras.layers.core import Flatten
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dropout, BatchNormalization

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
                # depth refers to RGB image
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.23))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (3 , 3), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.23))
                #3
		model.add(Conv2D(80, (3, 3), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.23))
                #4
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.23))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
