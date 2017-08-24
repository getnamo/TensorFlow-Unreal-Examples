#Converted to ue4 use from: https://www.tensorflow.org/get_started/mnist/beginners
#mnist_softmax.py: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import unreal_engine as ue
import upycmd as cmd
import sys
from TFPluginAPI import TFPluginAPI

import operator

class MnistSimple(TFPluginAPI):

	#expected api: setup your model for training
	def setup(self):
		#Setup our paths
		self.scripts_path = cmd.PythonProjectScriptPath()
		self.data_dir = cmd.AsAbsPath(self.scripts_path + '/dataset/mnist/')
		self.model_directory = cmd.AsAbsPath(self.scripts_path + "/model/mnistSimple")
		self.model_path = cmd.AsAbsPath(self.model_directory + "/model.ckpt")

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
				#debug errors
				#e = sys.exc_info()[0]
				#ue.log('train error: ' + str(e))

				W = tf.get_variable('W', [784, 10], initializer=tf.zeros_initializer)
				b = tf.get_variable('b', [10], initializer=tf.zeros_initializer)

				self.saver = tf.train.Saver()
				print("Initializing saver variables")

			#The rest of the operations are the same
			x = tf.placeholder(tf.float32, [None, 784])
			y = tf.matmul(x, W) + b

			#save the model model
			self.model = {'x':x, 'y':y, 'W':W,'b':b}

	#expected api: storedModel and session, json inputs
	def runJsonInput(self, jsonInput):
		#expect an image struct in json format
		pixelarray = jsonInput['pixels']
		ue.log('image len: ' + str(len(pixelarray)))

		#embedd the input image pixels as 'x'
		feed_dict = {self.model['x']: [pixelarray]}

		result = self.sess.run(self.model['y'], feed_dict)

		#convert our raw result to a prediction
		index, value = max(enumerate(result[0]), key=operator.itemgetter(1))

		ue.log('max: ' + str(value) + 'at: ' + str(index))

		#set the prediction result in our json
		jsonInput['prediction'] = index

		return jsonInput

	#expected api: no params forwarded for training? TBC
	def train(self):

		ue.log("mnist simple train")

		ue.log('Save/Load Variation, do we need to train?')

		#train model if we didn't find a trained model
		if not (self.model_loaded):
			with self.sess.as_default():
				with self.graph.as_default():
					#we just need to know our x and y, W and b are embedded in the y operation
					x = self.model['x']
					y = self.model['y']

					# Define loss and optimizer
					y_ = tf.placeholder(tf.float32, [None, 10])
					cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
					train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

					ue.log('No saved data found, starting training...')
					mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
					tf.global_variables_initializer().run()

					training_range = 1000
					# Train
					for i in range(training_range):
						batch_xs, batch_ys = mnist.train.next_batch(100)
						self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
						if i % 100 == 0:
							ue.log(i)
							if(self.shouldstop):
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

					#store optional summary information
					self.summary = {'x':str(x), 'y':str(y), 'W':str(self.model['W']), 'b':str(self.model['b'])}

		else:
			ue.log('Model already trained, skipping.')
			self.summary = {}
		

		self.stored['summary'] = self.summary
		return self.stored

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return MnistSimple.getInstance()
