import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

#utility imports
from random import randint
import collections
import numpy as np

class ExampleAPI(TFPluginAPI):

	#expected optional api: setup your model for training
	def onSetup(self):
		self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		self.x = tf.placeholder(tf.float32)
		
		#self.paddleY = tf.placeholder(tf.float32)
		#self.ballXY = tf.placeholder(tf.float32)
		#self.score = tf.placeholder(tf.float32)

		#use collections to manage a x frames buffer of input
		self.bufferLength = 200		
		self.inputQ = collections.deque(maxlen=self.bufferLength)

		#fill our deque so our input size is always the same
		for x in range(0, 199):
			self.inputQ.append([0.0,0.0,0.0,0.0])

		pass
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):
		
		#debug action
		action = randint(0,2)

		#layer our input using deque ~200 frames so we can train with temporal data 

		#make a 1D stack of current input
		ballPos = jsonInput['ballPosition']
		stackedInput = [jsonInput['paddlePosition'], ballPos['x'], ballPos['y'],jsonInput['actionScore']]

		#append our stacked input to our deque
		self.inputQ.append(stackedInput)

		#convert to list and set as x placeholder
		feed_dict = {self.x: list(self.inputQ)}

		#debug 
		#print(jsonInput)
		#print(stackedInput)
		#print(len(self.inputQ))	#deque should grow until max size
		#print(feed_dict)

		#build our input from our json values
		#feed_dict = {self.a: jsonInput['a'], self.b: jsonInput['b']}

		#rawResult = self.sess.run(self.c,feed_dict)

		#return random action
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