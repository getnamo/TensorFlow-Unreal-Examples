'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

#converted for ue4 use from
#https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

import json
from pathlib import Path

from tensorflow.python import keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K
import numpy as np
import operator
import sys
import random

import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

class MnistKeras(TFPluginAPI):

	#keras stop callback
	class StopCallback(keras.callbacks.Callback):
		def __init__(self, outer):
			self.outer = outer

		def on_train_begin(self, logs={}):
			self.losses = []

		def on_batch_end(self, batch, logs={}):
			if(self.outer.shouldStop):
				#notify on first call
				if not (self.model.stop_training):
					ue.log('Early stop called!')
				self.model.stop_training = True

			else:
				if(batch % 5 == 0):
					#json convertible types are float64 not float32
					logs['acc'] = np.float64(logs['acc'])
					logs['loss'] = np.float64(logs['loss'])
					self.outer.callEvent('TrainingUpdateEvent', logs, True)

				#callback an example image from batch to see the actual data we're training on
				if((batch*self.outer.batch_size) % 10000 == 0):
					index = random.randint(0,self.outer.batch_size)*batch
					self.outer.jsonPixels['pixels'] = self.outer.x_train[index].ravel().tolist()
					self.outer.callEvent('PixelEvent', self.outer.jsonPixels, True)

	#expected api: setup your model for training
	def onSetup(self):
		#setup or load your model and pass it into stored
		
		#Usually store session, graph, and model if using keras
		#self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		self.stopcallback = self.StopCallback(self)

	#expected api: storedModel and session, json inputs
	def onJsonInput(self, jsonInput):
		#build the result object
		result = {'prediction':-1}

		#prepare the input
		x_raw = [jsonInput['pixels']]
		x_raw = np.reshape(x_raw, (1, 28, 28))

		ue.log('image shape: ' + str(x_raw.shape))
		#ue.log(stored)

		#convert pixels to N_samples, height, width, N_channels input tensor
		x = np.reshape(x_raw, (len(x_raw), 28, 28, 1))

		ue.log('input shape: ' + str(x.shape))

		#run run the input through our network
		if self.model is None:
			ue.log("Warning! No 'model' found. Did training complete?")
			return result

		#restore our saved session and model
		K.set_session(self.session)

		with self.session.as_default():
			output = self.model.predict(x)

			ue.log(output)

			#convert output array to prediction
			index, value = max(enumerate(output[0]), key=operator.itemgetter(1))

			result['prediction'] = index
			result['pixels'] = jsonInput['pixels'] #unnecessary but useful for round tripping

		return result

	#expected api: no params forwarded for training? TBC
	def onBeginTraining(self):
		ue.log("starting mnist keras cnn training")

		model_file_name = "mnistKerasCNN"
		model_directory = ue.get_content_dir() + "/Scripts/"
		model_sess_path =  model_directory + model_file_name + ".tfsess"
		model_json_path = model_directory + model_file_name + ".json"

		my_file = Path(model_json_path)

		#reset the session each time we get training calls
		K.clear_session()

		#let's train
		batch_size = 128
		num_classes = 10
		epochs = 5 					 # lower default for simple testing
		self.batch_size = batch_size # so that it can be obtained inside keras callbacks

		# input image dimensions
		img_rows, img_cols = 28, 28

		# the data, shuffled and split between train and test sets
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
			x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
		else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
			x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255
		ue.log('x_train shape:' + str(x_train.shape))
		ue.log(str(x_train.shape[0]) + 'train samples')
		ue.log(str(x_test.shape[0]) + 'test samples')

		#pre-fill our callEvent data to optimize callbacks
		jsonPixels = {}
		size = {'x':28, 'y':28}
		jsonPixels['size'] = size
		self.jsonPixels = jsonPixels
		self.x_train = x_train

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

		model = Sequential()
		model.add(Conv2D(64, kernel_size=(3, 3),
						  activation='relu',
						  input_shape=input_shape))
		
		# model.add(Dropout(0.2))
		# model.add(Flatten())
		# model.add(Dense(512, activation='relu'))
		# model.add(Dropout(0.2))
		# model.add(Dense(num_classes, activation='softmax'))

		#model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adadelta(),
					  metrics=['accuracy'])

		model.fit(x_train, y_train,
				  batch_size=batch_size,
				  epochs=epochs,
				  verbose=1,
				  validation_data=(x_test, y_test),
				  callbacks=[self.stopcallback])
		score = model.evaluate(x_test, y_test, verbose=0)
		ue.log("mnist keras cnn training complete.")
		ue.log('Test loss:' + str(score[0]))
		ue.log('Test accuracy:' + str(score[1]))

		self.session = K.get_session()
		self.model = model

		stored = {'model':model, 'session': self.session}

		#run a test evaluation
		ue.log(x_test.shape)
		result_test = model.predict(np.reshape(x_test[500],(1,28,28,1)))
		ue.log(result_test)

		#flush the architecture model data to disk
		#with open(model_json_path, "w") as json_file:
		#	json_file.write(model.to_json())

		#flush the whole model and weights to disk
		#saver = tf.train.Saver()
		#save_path = saver.save(K.get_session(), model_sess_path)
		#model.save(model_path)

		
		return stored

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return MnistKeras.getInstance()
