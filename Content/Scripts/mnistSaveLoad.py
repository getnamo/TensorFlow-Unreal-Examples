#Converted to ue4 use from: https://www.tensorflow.org/get_started/mnist/beginners
#mnist_softmax.py: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import unreal_engine as ue
import sys
from TFPluginAPI import TFPluginAPI

import operator

class MnistSaveLoad(TFPluginAPI):

	#expected api: setup your model for training
	def onSetup(self):
		#Setup our paths
		self.scripts_path = ue.get_content_dir() + "Scripts"

		self.data_dir = self.scripts_path + '/dataset/mnist'
		self.model_directory = self.scripts_path + "/model/mnistSimple"
		self.model_path = self.model_directory + "/model.ckpt"

		#startup a session and try to obtain latest save
		self.sess = tf.InteractiveSession()
		self.graph = tf.get_default_graph()
		self.model_loaded = False

		with self.sess.as_default():
			try:
				saver = tf.train.import_meta_graph(self.model_path + ".meta")
				ue.log('meta graph imported')
				saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))
				ue.log('graph restored')
				self.model_loaded = True

				#restore our weights
				self.graph = tf.get_default_graph()
				W = self.graph.get_tensor_by_name("W:0")
				b = self.graph.get_tensor_by_name("b:0")

			except:
				#no saved model, initialize our variables
				W = tf.get_variable('W', [784, 10], initializer=tf.zeros_initializer)
				b = tf.get_variable('b', [10], initializer=tf.zeros_initializer)

				print("no model saved, initializing variables")

			self.saver = tf.train.Saver()
				
			#The rest of the operations are the same
			x = tf.placeholder(tf.float32, [None, 784])
			y = tf.matmul(x, W) + b

			#store the model in a class instance variable to easily reference in another function
			self.model = {'x':x, 'y':y, 'W':W,'b':b}

	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):
		#expect an image struct in json format
		pixelarray = jsonInput['pixels']
		ue.log('image len: ' + str(len(pixelarray)))

		#map the input image pixels to the 'x' placeholder
		feed_dict = {self.model['x']: [pixelarray]}

		#run the input feed_dict through the model 'y' and obtain a result
		result = self.sess.run(self.model['y'], feed_dict)

		#convert our raw result to a prediction by picking the highest value from 1D result tensor
		index, value = max(enumerate(result[0]), key=operator.itemgetter(1))

		ue.log('max: ' + str(value) + 'at: ' + str(index))

		#set the prediction result in our json
		jsonInput['prediction'] = index

		return jsonInput

	#expected optional api: start training your network
	def onBeginTraining(self):

		ue.log("mnist simple train")

		ue.log('Save/Load Variation, do we need to train?')

		#train model if we didn't find a trained model or we're forcing retraining
		if (not self.model_loaded) or (self.shouldRetrain):
			ue.log('No saved data found, starting training...')

			with self.sess.as_default():
				with self.graph.as_default():
					#we just need to know our x and y, W and b are embedded in the y operation
					x = self.model['x']
					y = self.model['y']

					# Define loss and optimizer
					y_ = tf.placeholder(tf.float32, [None, 10])
					cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
					train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

					#read in mnist data to use to feed x during training
					mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
					tf.global_variables_initializer().run()

					training_range = 1000
					# Train
					for i in range(training_range):
						batch_xs, batch_ys = mnist.train.next_batch(100)

						self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
						if i % 100 == 0:
							ue.log(i)
							if(self.shouldStop):
								ue.log('early break')
								break 

					# Test trained model
					correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
					accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
					finalAccuracy = self.sess.run(accuracy, feed_dict={x: mnist.test.images,
													  y_: mnist.test.labels})
					ue.log('final training accuracy: ' + str(finalAccuracy))


					#save our trained variables
					save_path = self.saver.save(self.sess, self.model_path) #, global_step=training_range)
					print("Model saved in file: %s" % save_path)

					#update model in memory
					self.model['x'] = x
					self.model['y'] = y

					#Optional: append summary information
					self.summary = {'x':str(x), 'y':str(y), 'W':str(self.model['W']), 'b':str(self.model['b'])}

		else:
			ue.log('Model already trained, skipping.')
			
			#Optional: store an empty summary variable
			self.summary = {}
		
		#Optional: append a summary object to our self.stored that we return on from the training function
		self.stored['summary'] = self.summary
		return self.stored

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return MnistSaveLoad.getInstance()
