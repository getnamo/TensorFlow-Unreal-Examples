import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

class ExampleAPI(TFPluginAPI):

	#expected optional api: setup your model for training
	def onSetup(self):
		self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		self.a = tf.placeholder(tf.float32)
		self.b = tf.placeholder(tf.float32)

		#operation
		self.c = self.a + self.b
		pass
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):
		
		print(jsonInput)

		feed_dict = {self.a: jsonInput['a'], self.b: jsonInput['b']}

		rawResult = self.sess.run(self.c,feed_dict)

		return {'c':rawResult.tolist()}

	#custom function to change the op
	def changeOperation(self, type):
		if(type == '+'):
			self.c = self.a + self.b

		elif(type == '-'):
			self.c = self.a - self.b


	#expected optional api: start training your network
	def onBeginTraining(self):
		pass
    
#NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return ExampleAPI.getInstance()