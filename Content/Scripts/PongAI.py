import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

#utility imports
from random import randint
import collections
import numpy as np

#part of structure taken from https://gist.github.com/arushir/04c58283d4fc00a4d6983dc92a3f1021
from dqn import DQN

class ExampleAPI(TFPluginAPI):

	#expected optional api: setup your model for training
	def onSetup(self):
		self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		self.x = tf.placeholder(tf.float32)
		
		#self.paddleY = tf.placeholder(tf.float32)
		#self.ballXY = tf.placeholder(tf.float32)
		#self.score = tf.placeholder(tf.float32)

		self.num_actions = 3

		DEFAULT_EPISODES = 2000
		DEFAULT_STEPS = 500 
		DEFAULT_ENVIRONMENT = 'Pong-UE4'

		DEFAULT_MEMORY_CAPACITY = 10000
		DEFAULT_EPSILON = 0.1
		DEFAULT_GAMMA = 0.9
		DEFAULT_MINI_BATCH_SIZE = 10

		DEFAULT_LEARNING_RATE = 0.0001
		DEFAULT_REGULARIZATION = 0.001
		DEFAULT_NUM_HIDDEN = 2 # not used in tensorflow implementation
		DEFAULT_HIDDEN_SIZE = 20

		self.agent_params = {'episodes': DEFAULT_EPISODES, 'steps': DEFAULT_STEPS, 'environment': DEFAULT_ENVIRONMENT, 'run_id': 1}
		self.cnn_params = {'lr': DEFAULT_LEARNING_RATE, 'reg': DEFAULT_REGULARIZATION, 'num_hidden':DEFAULT_NUM_HIDDEN,'hidden_size':DEFAULT_HIDDEN_SIZE,'mini_batch_size': DEFAULT_MINI_BATCH_SIZE}
		self.dqn_params = {'memory_capacity':DEFAULT_MEMORY_CAPACITY, 'epsilon':DEFAULT_EPSILON, 'gamma':DEFAULT_GAMMA,'mini_batch_size':DEFAULT_MINI_BATCH_SIZE}

		#use collections to manage a x frames buffer of input
		self.memory_capacity = 200
		self.inputQ = collections.deque(maxlen=self.memory_capacity)

		null_input = np.zeros(4)
		self.observation_shape = null_input.shape
		self.model = DQN(self.num_actions, self.observation_shape, self.dqn_params, self.cnn_params)

		#fill our deque so our input size is always the same
		for x in range(0, self.memory_capacity-1):
			self.inputQ.append(null_input)

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