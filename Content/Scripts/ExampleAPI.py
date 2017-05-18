import sys

import tensorflow as tf
import unreal_engine as ue

#plugin scoped variable for stopping
this = sys.modules[__name__]
this.shouldstop = False

#expected api: storedModel and session, json inputs
def runJsonInput(stored, jsonInput):
	#e.g. our json input could be a pixel array
	#pixelarray = jsonInput['pixels']

	#run input on your graph
	#e.g. sess.run(model['y'], feed_dict)
	# where y is your result graph and feed_dict is {x:[input]}

	#...

	#return a json you will parse e.g. a prediction
	result = {}
	result['prediction'] = 0

	return result

#expected api: no params forwarded for training? TBC
def train():

	#Usually store session, graph, and model if using keras
	sess = tf.InteractiveSession()

	#train here

	#...

	#inside your training loop check if we should stop early
	#if(this.shouldstop):
	#	break

	stored = {'session': sess, 'graph': tf.get_default_graph()}

	#return an object with store parameters, these will be sent to runJsonInput
	return stored

#expected api: early stopping
def stop():
	this.shouldstop = True