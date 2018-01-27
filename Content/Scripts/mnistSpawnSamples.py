#Converted to ue4 use from: https://www.tensorflow.org/get_started/mnist/beginners
#mnist_softmax.py: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

import operator

class MnistSimple(TFPluginAPI):
	
	#expected api: storedModel and session, json inputs
	def onJsonInput(self, jsonInput):
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
	def onBeginTraining(self):

		ue.log("starting mnist simple training")

		self.scripts_path = ue.get_content_dir() + "Scripts"
		self.data_dir = self.scripts_path + '/dataset/mnist'

		mnist = input_data.read_data_sets(self.data_dir, one_hot=True)

		# Create the model
		x = tf.placeholder(tf.float32, [None, 784])
		W = tf.Variable(tf.zeros([784, 10]))
		b = tf.Variable(tf.zeros([10]))
		y = tf.matmul(x, W) + b

		# Define loss and optimizer
		y_ = tf.placeholder(tf.float32, [None, 10])

		# The raw formulation of cross-entropy,
		#
		#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
		#                                 reduction_indices=[1]))
		#
		# can be numerically unstable.
		#
		# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
		# outputs of 'y', and then average across the batch.
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

		#update session for this thread
		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		training_range = 1000

		#pre-fill our callEvent data to minimize setting
		jsonPixels = {}
		size = {'x':28, 'y':28}
		jsonPixels['size'] = size

		# Train
		for i in range(training_range):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			self.sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			if i % 100 == 0:
				ue.log(i)

				#send two pictures from our dataset per batch
				jsonPixels['pixels'] = batch_xs[0].tolist()
				self.callEvent('PixelEvent', jsonPixels, True)
				jsonPixels['pixels'] = batch_xs[49].tolist()
				self.callEvent('PixelEvent', jsonPixels, True)

				if(self.shouldStop):
					ue.log('early break')
					break 

		# Test trained model
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		finalAccuracy = self.sess.run(accuracy, feed_dict={x: mnist.test.images,
										  y_: mnist.test.labels})
		ue.log('final training accuracy: ' + str(finalAccuracy))
		
		#return trained model
		self.model = {'x':x, 'y':y, 'W':W,'b':b}

		#store optional summary information
		self.summary = {'x':str(x), 'y':str(y), 'W':str(W), 'b':str(b)}

		self.stored['summary'] = self.summary
		return self.stored

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return MnistSimple.getInstance()
