#converted for ue4 use from
#https://github.com/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb

import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

#additional includes
from tensorflow.python.keras import backend as K	#to ensure things work well with multi-threading
import numpy as np   	#for reshaping input
import operator      	#used for getting max prediction from 1x10 output array

class MnistTutorial(TFPluginAPI):

	#Called when TensorflowComponent sends Json input
	def onJsonInput(self, jsonInput):
		#build the result object
		result = {'prediction':-1}

		#If we try to predict before training is complete
		if not hasattr(self, 'model'):
			ue.log_warning("Warning! No 'model' found, prediction invalid. Did training complete?")
			return result

		#prepare the input, reshape 784 array to a 1x28x28 array
		x_raw = jsonInput['pixels']
		x = np.reshape(x_raw, (1, 28, 28))

		#run the input through our network using stored model and graph
		with self.graph.as_default():
			output = self.model.predict(x)

			#convert output array to max value prediction index (0-10)
			index, value = max(enumerate(output[0]), key=operator.itemgetter(1))

			#Optionally log the output so you can see the weights for each value and final prediction
			ue.log('Output array: ' + str(output) + ',\nPrediction: ' + str(index))

			result['prediction'] = index

		return result

	#Called when TensorflowComponent signals begin training (default: begin play)
	def onBeginTraining(self):
		ue.log("starting MnistTutorial training")

		#reset the session each time we get training calls
		K.clear_session()

		#load mnist data set
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		#rescale 0-255 -> 0-1.0
		x_train, x_test = x_train / 255.0, x_test / 255.0

		#define model
		model = tf.keras.models.Sequential([
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(512, activation=tf.nn.relu),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(10, activation=tf.nn.softmax)
		])

		model.compile(	optimizer='adam',
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])

		#this will do the actual training
		model.fit(x_train, y_train, epochs=1)
		model.evaluate(x_test, y_test)

		ue.log("Training complete.")

		#store our model and graph for prediction
		self.graph = tf.get_default_graph()
		self.model = model

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return MnistTutorial.getInstance()
