import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI
from random import randint

class ExampleAPI(TFPluginAPI):

	#expected optional api: setup your model for training
	def onSetup(self):
		self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		#operation
		#self.c = self.a + self.b
		pass
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):
		
		#action = randint(0,2)

		ballPos = jsonInput['ballPosition']
		paddleY = jsonInput['paddlePosition']
		ballY = ballPos['y']

		#track the ball
		if(paddleY < ballY):
			action = 1
		elif(paddleY > ballY):
			action = 2
		else:
			action = 0

		#debug
		#print(jsonInput)
		#print(action)

		#just do a random action
		return {'action':action}

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