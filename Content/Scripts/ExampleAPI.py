import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

class ExampleAPI(TFPluginAPI):

	#expected api: setup your model for training
	def setup(self):
		#setup or load your model and pass it into stored
		
		#Usually store session, graph, and model if using keras
		self.sess = tf.InteractiveSession()
		self.graph = tf.get_default_graph()

	#expected api: storedModel and session, json inputs
	def runJsonInput(self, jsonInput):
		#e.g. our json input could be a pixel array
		#pixelarray = jsonInput['pixels']

		#run input on your graph
		#e.g. sess.run(model['y'], feed_dict)
		# where y is your result graph and feed_dict is {x:[input]}

		#...

		#return a json you will parse e.g. a prediction
		result = {}
		result['prediction'] = -1

		return result

	#expected api: no params forwarded for training? TBC
	def train(self):
		#train here

		#...

		#inside your training loop check if we should stop early
		#if(this.shouldstop):
		#	break
		pass

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return ExampleAPI.getInstance()