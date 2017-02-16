from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import operator

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import unreal_engine as ue

def main(data_dir):
	mnist = input_data.read_data_sets(data_dir, one_hot=True)

	ue.log("starting mnist simple")

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
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	# Train
	tf.global_variables_initializer().run()
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		if i % 100 == 0: 
			ue.log(i)

	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	finalAccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
									  y_: mnist.test.labels})
	ue.log('final training accuracy: ' + str(finalAccuracy))
	
	#return trained model
	stored = {'x':x, 'y':y, 'W':W,'b':b, 'session':sess}

	#ue.log('trained x: ' + str(stored['x']))
	#ue.log('trained y: ' + str(stored['y']))
	#ue.log('trained W: ' + str(stored['W']))
	#ue.log('trained b: ' + str(stored['b']))

	#store optional summary information
	stored['summary'] = {'x':str(x), 'y':str(y), 'W':str(W), 'b':str(b)}

	return stored

def runModelWithInput(session, model, input_dict):
	return session.run(model['y'], input_dict)

#expected api: storedModel and session, json inputs
def runJsonInput(stored, jsonInput):
	#expect an image struct in json format
	pixelarray = jsonInput['pixels']
	ue.log('image len: ' + str(len(pixelarray)))

	#embedd the input image pixels as 'x'
	feed_dict = {stored['x']: [pixelarray]}

	ue.log(feed_dict)

	#run our model
	result = runModelWithInput(stored['session'], stored, feed_dict)
	ue.log('result: ' + str(len(pixelarray)))

	#convert our raw result to a prediction
	index, value = max(enumerate(result[0]), key=operator.itemgetter(1))

	ue.log('max: ' + str(value) + 'at: ' + str(index))

	#set the prediction result in our json
	jsonInput['prediction'] = index

	return jsonInput

#expected api: no params forwarded for training? TBC
def train():
	data_dir = '/tmp/tensorflow/mnist/input_data'
	return main(data_dir)