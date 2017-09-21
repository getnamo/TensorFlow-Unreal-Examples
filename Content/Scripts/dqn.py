import numpy as np
import random as random
from collections import deque

from cnn_tensorflow import CNN

# See https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf for model description

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