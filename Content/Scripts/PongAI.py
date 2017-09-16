import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

class ExampleAPI(TFPluginAPI):

	#expected optional api: setup your model for training
	def onSetup(self):
		self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		self.paddleY = tf.placeholder(tf.float32)
		self.ballXY = tf.placeholder(tf.float32)
		self.score = tf.placeholder(tf.float32)

		#operation
		#self.c = self.a + self.b
		pass
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):
		
		print(jsonInput)

		#build our input from our json values
		#feed_dict = {self.a: jsonInput['a'], self.b: jsonInput['b']}

		#rawResult = self.sess.run(self.c,feed_dict)

		#return do nothing action for now
		return {'action':0}

	#custom function to determine which paddle we are
	def setPaddleType(self, type):
		self.paddle = 0
		if(type == 'PaddleRight'):
			self.paddle = 1


	#expected optional api: start training your network
	def onBeginTraining(self):
		pass
    
#NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return ExampleAPI.getInstance()