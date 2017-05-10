'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import unreal_engine as ue
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.keras.api.keras.models import Sequential, load_model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import model_from_json
from tensorflow.contrib.keras import backend as K
import numpy as np
import operator

#expected api: storedModel and session, json inputs
def runJsonInput(stored, jsonInput):
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
	if stored is None:
		ue.log("Warning! No 'stored' found. Did training complete?")
		return result

	#restore our saved session and model
	K.set_session(stored['session'])
	graph = stored['graph']
	
	with graph.as_default():
		output = stored['model'].predict(x)

		ue.log(output)

		#convert output array to prediction
		index, value = max(enumerate(output[0]), key=operator.itemgetter(1))

		result['prediction'] = index;
		result['pixels'] = jsonInput['pixels'] #unnecessary but useful for round tripping

		return result

#debug
# def notrain():
# 	#not ready yet
# 	ue.log("starting mnist keras cnn training")

# 	path = ue.get_content_dir() +"model.json"

# 	ue.log(path)
	
# 	with open(path, "w") as json_file:
# 		json_file.write(model_json)

# 	json_file = open(path, 'r')
# 	loaded_model_json = json_file.read()
# 	ue.log("read data: " + loaded_model_json)

# 	ue.log('write test complete')

#expected api
def train():

	ue.log("starting mnist keras cnn training")

	model_file_name = "mnistKerasCNN"
	model_directory = ue.get_content_dir() + "/Scripts/"
	model_sess_path =  model_directory + model_file_name + ".tfsess"
	model_json_path = model_directory + model_file_name + ".json"

	my_file = Path(model_json_path)

	#reset the session each time we get training calls
	K.clear_session()

	#todo: fix saving...
	if False: #my_file.is_file():
		ue.log("reading model file: " + model_json_path)
		
		model_file_r = open(model_json_path, 'r')
		loaded_model_json = model_file_r.read()

		if loaded_model_json is None:
			ue.log('No model found in file, proceeding to training')
		else:
			ue.log('Model is already trained, done.')
			
			#saver = tf.train.Saver()

			#sess = K.get_session()
			#saver.restore(sess, model_sess_path)

			#load the model and session
			stored = {'model':model_from_json(loaded_model_json), 'session': K.get_session()}

			return stored
	else:
		ue.log("no model file found")


	#let's train
	batch_size = 128
	num_classes = 10
	epochs = 1

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

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
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
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	ue.log("mnist keras cnn training complete.")
	ue.log('Test loss:' + str(score[0]))
	ue.log('Test accuracy:' + str(score[1]))

	#stored object, set it to however you like, you'll use input that you define
	#stored = {}
	#stored['model'] = model
	#stored['session'] = K.get_session()

	stored = {'model':model, 'session': K.get_session(), 'graph': tf.get_default_graph()}

	#run a test evaluation
	#ue.log(x_test.shape)
	#result_test = model.predict(np.reshape(x_test[0],(1,28,28,1)))
	#ue.log(result_test)

	#flush the architecture model data to disk
	#with open(model_json_path, "w") as json_file:
	#	json_file.write(model.to_json())

	#flush the whole model and weights to disk
	#saver = tf.train.Saver()
	#save_path = saver.save(K.get_session(), model_sess_path)
	#model.save(model_path)

	return stored

