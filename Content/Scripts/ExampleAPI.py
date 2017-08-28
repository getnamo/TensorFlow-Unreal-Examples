import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

class ExampleAPI(TFPluginAPI):

	#expected api: setup your model for your use cases
	def onSetup(self):
		#setup or load your model and pass it into stored
		
		#Usually store session, graph, and model if using keras
		self.sess = tf.InteractiveSession()
		self.graph = tf.get_default_graph()

	#expected api: storedModel and session, json inputs
	def onJsonInput(self, jsonInput):
		#e.g. our json input could be a pixel array
		#pixelarray = jsonInput['pixels']

		#run input on your graph
		#e.g. sess.run(model['y'], feed_dict)
		# where y is your result graph and feed_dict is {x:[input]}

		#...

		#you can also call an event e.g.
		#callEvent('myEvent', 'myData')

		#return a json you will parse e.g. a prediction
		result = {}
		result['prediction'] = -1

		return result

	#optional api: no params forwarded for training? TBC
	def onBeginTraining(self):
		#train here

		#...

		#inside your training loop check if we should stop early
		#if(this.shouldStop):
		#	break
		pass

	#optional api: use if you need some things to happen if we get stopped
	def onStopTraining(self):
		#you should be listening to this.shouldstop, but you can also receive this call
		pass

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return ExampleAPI.getInstance()