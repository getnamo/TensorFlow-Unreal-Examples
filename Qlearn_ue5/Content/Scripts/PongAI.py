import tensorflow as tf
import unreal_engine as ue
from mlpluginapi import MLPluginAPI

#utility imports
from random import randint
import collections
import numpy as np
import logging
import random as random
from collections import deque

class CNN:
  """
  Convolutional Neural Network model.
  """

  def __init__(self, num_actions, observation_shape, params={}, verbose=False):
    """
    Initialize the CNN model with a set of parameters.
    Args:
      params: a dictionary containing values of the models' parameters.
    """

    self.verbose = verbose
    self.num_actions = num_actions

    # observation shape will be a tuple
    self.observation_shape = observation_shape[0]
    logging.info('Initialized with params: {}'.format(params))

    self.lr = params['lr']
    self.reg = params['reg']
    self.num_hidden = params['num_hidden']
    self.hidden_size = params['hidden_size']

    self.session = self.create_model()


  def add_placeholders(self):
    input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

    return input_placeholder, labels_placeholder, actions_placeholder


  def nn(self, input_obs):
    with tf.name_scope("Layer1") as scope:
      W1shape = [self.observation_shape, self.hidden_size]
      W1 = tf.get_variable("W1", shape=W1shape,)
      bshape = [1, self.hidden_size]
      b1 = tf.get_variable("b1", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("Layer2") as scope:
      W2shape = [self.hidden_size, self.hidden_size]
      W2 = tf.get_variable("W2", shape=W2shape,)
      bshape = [1, self.hidden_size]
      b2 = tf.get_variable("b2", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("OutputLayer") as scope:
      Ushape = [self.hidden_size, self.num_actions]
      U = tf.get_variable("U", shape=Ushape)
      b3shape = [1, self.num_actions]
      b3 = tf.get_variable("b3", shape=b3shape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(input_obs, W1)
    h = tf.tanh(tf.add(xW, b1))

    xW = tf.matmul(h, W2)
    h = tf.tanh(tf.add(xW, b2))

    hU = tf.matmul(h, U)    
    out = tf.add(hU, b3)

    reg = self.reg * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(U)))
    return out, reg


  def create_model(self):
    """
    The model definition.
    """
    self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()
    outputs, reg = self.nn(self.input_placeholder)
    self.predictions = outputs
    
    self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)

    self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)

    self.train_op = optimizer.minimize(self.loss)
    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    return session

  def train_step(self, Xs, ys, actions):
    """
    Updates the CNN model with a mini batch of training examples.
    """

    loss, _, prediction_probs, q_values = self.session.run(
      [self.loss, self.train_op, self.predictions, self.q_vals],
      feed_dict = {self.input_placeholder: Xs,
                  self.labels_placeholder: ys,
                  self.actions_placeholder: actions
                  })

  def predict(self, observation):
    """
    Predicts the rewards for an input observation state. 
    Args:
      observation: a numpy array of a single observation state
    """

    loss, prediction_probs = self.session.run(
      [self.loss, self.predictions],
      feed_dict = {self.input_placeholder: observation,
                  self.labels_placeholder: np.zeros(len(observation)),
                  self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                  })

    return prediction_probs
class DQN:
  def __init__(self, num_actions, observation_shape, dqn_params, cnn_params):
    self.num_actions = num_actions
    self.epsilon = dqn_params['epsilon']
    self.gamma = dqn_params['gamma']
    self.mini_batch_size = dqn_params['mini_batch_size']

    # memory 
    self.memory = deque(maxlen=dqn_params['memory_capacity'])

    # initialize network
    self.model = CNN(num_actions, observation_shape, cnn_params)
    print("model initialized")

  def select_action(self, observation):
    """
    Selects the next action to take based on the current state and learned Q.
    Args:
      observation: the current state
    """

    if random.random() < self.epsilon: 
      # with epsilon probability select a random action 
      action = np.random.randint(0, self.num_actions)
    else:
      # select the action a which maximizes the Q value
      obs = np.array([observation])
      q_values = self.model.predict(obs)
      action = np.argmax(q_values)

    return action

  def update_state(self, action, observation, new_observation, reward, done):
    """
    Stores the most recent action in the replay memory.
    Args: 
      action: the action taken 
      observation: the state before the action was taken
      new_observation: the state after the action is taken
      reward: the reward from the action
      done: a boolean for when the episode has terminated 
    """
    transition = {'action': action,
                  'observation': observation,
                  'new_observation': new_observation,
                  'reward': reward,
                  'is_done': done}
    self.memory.append(transition)

  def get_random_mini_batch(self):
    """
    Gets a random sample of transitions from the replay memory.
    """
    rand_idxs = random.sample(range(len(self.memory)), self.mini_batch_size)
    mini_batch = []
    for idx in rand_idxs:
      mini_batch.append(self.memory[idx])

    return mini_batch

  def train_step(self):
    """
    Updates the model based on the mini batch
    """
    if len(self.memory) > self.mini_batch_size:
      mini_batch = self.get_random_mini_batch()

      Xs = []
      ys = []
      actions = []

      for sample in mini_batch:
        y_j = sample['reward']

        # for nonterminals, add gamma*max_a(Q(phi_{j+1})) term to y_j
        if not sample['is_done']:
          new_observation = sample['new_observation']
          new_obs = np.array([new_observation])
          q_new_values = self.model.predict(new_obs)
          action = np.max(q_new_values)
          y_j += self.gamma*action

        action = np.zeros(self.num_actions)
        action[sample['action']] = 1

        observation = sample['observation']

        Xs.append(observation.copy())
        ys.append(y_j)
        actions.append(action.copy())

      Xs = np.array(Xs)
      ys = np.array(ys)
      actions = np.array(actions)

      self.model.train_step(Xs, ys, actions)

#part of structure taken from https://gist.github.com/arushir/04c58283d4fc00a4d6983dc92a3f1021
#from dqn import DQN

class ExampleAPI(MLPluginAPI):

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
		self.actionQ = collections.deque(maxlen=self.memory_capacity)

		null_input = np.zeros(3)
		self.observation_shape = null_input.shape
		#self.model = DQN(self.num_actions, self.observation_shape, self.dqn_params, self.cnn_params)

		#fill our deque so our input size is always the same
		for x in range(0, self.memory_capacity):
			self.inputQ.append(null_input)
			self.actionQ.append(0)

		pass
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):
		
		#debug action
		action = randint(0,2)
        	#track the ball
		if(paddleY < ballY):
			action = 1
		elif(paddleY > ballY):
			action = 2
		else:
			action = 1
          #ue.log(action)

		#layer our input using deque ~200 frames so we can train with temporal data 

		#make a 1D stack of current input
		ballPos = jsonInput['ballPosition']
		observation = [jsonInput['paddlePosition'], ballPos['x'], ballPos['y']]
		reward = jsonInput['actionScore']

		#convert to list and set as x placeholder
		#feed_dict = {self.x: stackedList}
		#new_observation, reward, done, _ = env.step(action)

		#print(len(self.actionQ))
		lastAction = self.actionQ[self.memory_capacity-1]
		lastObservation = self.inputQ[self.memory_capacity-1]
		done = False
		
		# update the state 
		self.model.update_state(lastAction, lastObservation, observation, reward, done)

		# train step
		self.model.train_step()

		#append our stacked input to our deque
		self.inputQ.append(observation)
		#stackedList = list(self.inputQ)

		action = self.model.select_action(observation)
		self.actionQ.append(action)

		#debug 
		#print(jsonInput)
		#print(stackedInput)
		print(jsonInput['actionScore'])
		#print(len(self.inputQ))	#deque should grow until max size
		#print(feed_dict)

		#return selected action
		return {'action':float(action)}

	#custom function to determine which paddle we are
	def setPaddleType(self, type):
		self.paddle = 0
		if(type == 'PaddleRight'):
			self.paddle = 1

		ballPos = jsonInput['ballPosition']
		paddleY = jsonInput['paddlePosition']
		ballY = ballPos['y']

	
              

	#expected optional api: start training your network
	def onBeginTraining(self):
		pass
	
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
		
		action = randint(0,2)

		#debug
		#ue.log(jsonInput)
		ue.log(action)

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
def get_api():
	#return CLASSNAME.getInstance()
	return ExampleAPI.get_instance()