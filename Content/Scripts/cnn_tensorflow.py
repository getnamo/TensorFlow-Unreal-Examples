import tensorflow as tf
import numpy as np
import logging

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