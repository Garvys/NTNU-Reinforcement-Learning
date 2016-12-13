import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from copy import deepcopy
from policy import *
from replay_memory import *
from prioritized_replay_memory import *
from dueling_config import *
from bootstrapping_config import *
from q_learning_config import *
from replay_memory_config import *		
		
		
##
## @brief      Class implementing the Double Q Learning algorithm. Also Dueling + Bootstrapping + Prioritized Experience Replay
##
class DoubleDQNAgent:

	##
	## @brief      Constructs the object.
	##
	## @param      self                  The object
	## @param      action_space          The action space
	## @param      observation_space     The observation space
	## @param      main_network_layers   A list with the number of neurons in each layer
	## @param      q_learning_config     The S learning configuration
	## @param      replay_memory_config  The replay memory configuration
	## @param      bootstrapping_config  The bootstrapping configuration
	## @param      dueling_config        The dueling configuration
	## @param      policy                The policy
	##
	def __init__(self, action_space, observation_space, main_network_layers = [200],
		q_learning_config = QLearningConfig(), replay_memory_config = ReplayMemoryConfig(),
		bootstrapping_config = BootstrappingConfig(), dueling_config = DuelingConfig(),
		policy = EpsilonGreedyPolicy(0.3)):

		#Policy to use for the training
		self.policy = policy

		#Hyper parameters for the Qlearning part
		self.q_learning_config = q_learning_config

		#Hyper parameters for the Dueling Part
		self.dueling_config = dueling_config

		#Hyper parameters for the Bootstrapping part
		self.bootstrapping_config = bootstrapping_config

		#Replay memory
		self.replay_memory_config = replay_memory_config
		self.memory = {}
		if self.replay_memory_config.use_prioritized_replay:
			self.memory['A'] = PrioritizedReplayMemory(replay_memory_config.memorySize, replay_memory_config, learn_start = q_learning_config.size_replay_min_to_train, batch_size = q_learning_config.batch_size)
			self.memory['B'] = PrioritizedReplayMemory(replay_memory_config.memorySize, replay_memory_config, learn_start = q_learning_config.size_replay_min_to_train, batch_size = q_learning_config.batch_size)
		else:
			self.memory['A'] = ReplayMemory(replay_memory_config.memorySize, q_learning_config.batch_size)
			self.memory['B'] = ReplayMemory(replay_memory_config.memorySize, q_learning_config.batch_size)

		self.main_network_layers = main_network_layers

		#Intern variables
		self._time = 0
		self.learning_rate = q_learning_config.learning_rate_start
		self.head_used = 0
		self._dim_state = observation_space.shape[0]
		self._nb_actions = action_space.n
		self._action_space = action_space
		self.initializer = tf.truncated_normal_initializer(0, 0.02)

		#Build training Q network

		#Input
		self._tf_state = {}
		self._tf_action_mask = {}
		self._tf_y = {}
		self._tf_w = {}
		self._tf_learning_rate = {}

		#Output
		self._tf_action_values = {}
		self._tf_loss = {}
		self._tf_training = {}
		self._tf_td_error_per_sample = {}

		#Create Q_A
		self.build_q_network('A')
		#Create Q_B
		self.build_q_network('B')

		self._tf_session = tf.InteractiveSession()

		self._tf_session.run(tf.initialize_all_variables())

	##
	## @brief      Update the value of the learning rate over time
	##
	## @param      self  The object
	##
	def update_learning_rate(self):

		a = (self.q_learning_config.learning_rate_start - self.q_learning_config.learning_rate_end) / (0 - self.q_learning_config.time_learning_rate_end)
		b = self.q_learning_config.learning_rate_start

		lr = a*self._time + b
		self.learning_rate = max(lr, self.q_learning_config.learning_rate_end)

	##
	## @brief      Create a TensorfloW Q Network
	##
	## @param      self  The object
	## @param      name  The name
	##
	## @return     The quarter network.
	##
	def build_q_network(self, name):

		#State to predict the action values
		state = tf.placeholder(tf.float32, [None, self._dim_state])

		#Action mask
		action_mask = tf.placeholder(tf.float32, [None, self._nb_actions])

		#Effective reward
		y = tf.placeholder(tf.float32, [None, ])

		#Core network
		_net = self.create_network(state, 'fc_' + name)

		####### Duelling + Bootstrapping

		t = _net / self.bootstrapping_config.nb_heads
		net = t + tf.stop_gradient(_net - t)

		tab_losses = []
		tab_action_values = []
		tab_td_errors = []

		for i in range(self.bootstrapping_config.nb_heads):
			action_values = None
			if self.dueling_config.use_dueling:

				t_value_hidden = slim.fully_connected(net, self.dueling_config.size_net_value, activation_fn=None, scope='value_hidden_' + name + '_head_' + str(i), 
					weights_initializer=self.initializer, biases_initializer=self.initializer)

				t_value = slim.fully_connected(t_value_hidden, 1, activation_fn=None, scope='value_' + name + '_head_' + str(i), 
					weights_initializer=self.initializer, biases_initializer=self.initializer)

				t_adv_hidden = slim.fully_connected(net, self.dueling_config.size_net_adv, activation_fn=None, scope='adv_hidden_' + name + '_head_' + str(i), 
					weights_initializer=self.initializer, biases_initializer=self.initializer)

				t_adv = slim.fully_connected(t_adv_hidden, self._nb_actions, activation_fn=None, scope='adv_' + name + '_head_' + str(i), 
					weights_initializer=self.initializer, biases_initializer=self.initializer)

				#######

				#Output layer
				action_values = t_value + (t_adv - tf.reduce_mean(t_adv, reduction_indices=1, keep_dims=True))
			else:
				fc_head = net
				if self.bootstrapping_config.nb_heads > 1:
					fc_head = slim.fully_connected(net, self.bootstrapping_config.size_layer_head, activation_fn=None, scope='fc_' + name + '_head_' + str(i), 
						weights_initializer=self.initializer, biases_initializer=self.initializer)
				action_values = slim.fully_connected(fc_head, self._nb_actions, activation_fn=None, scope='qvalues_' + name + '_head_' + str(i), 
					weights_initializer=self.initializer, biases_initializer=self.initializer)

			#Q(s_j,a_j)
			q_predicted = tf.reduce_sum(tf.mul(action_mask, action_values), reduction_indices=1)

			td_error = y - q_predicted
			loss = tf.square(td_error)

			tab_action_values.append(action_values)

			tab_td_errors.append(td_error)
			tab_losses.append(loss)

		total_td_errors_per_sample = tf.reduce_sum(tab_td_errors, reduction_indices = 0)
		total_losses_per_sample = tf.reduce_sum(tab_losses, reduction_indices = 0)

		######

		x = total_losses_per_sample
		w = tf.placeholder(tf.float32, [None,])
		t_bis = tf.mul(x, w)
		y_bis = t_bis + tf.stop_gradient(x - t_bis)

		total_loss = tf.reduce_sum(y_bis)

		learning_rate = tf.placeholder(tf.float32, shape=[])
		training = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

		#Input
		self._tf_state[name] = state
		self._tf_action_mask[name] = action_mask
		self._tf_y[name] = y 
		self._tf_w[name] = w
		self._tf_learning_rate[name] = learning_rate

		#Output
		self._tf_action_values[name] = tab_action_values
		self._tf_loss[name] = total_loss
		self._tf_training[name] = training
		self._tf_td_error_per_sample[name] = total_td_errors_per_sample

	##
	## @brief      Create the main network
	##
	## @param      self        The object
	## @param      inputs      The inputs
	## @param      name_scope  The name scope
	## @param      reuse       Shared variables ?
	##
	## @return     Tf neural network
	##
	def create_network(self, inputs, name_scope, reuse = False):
		net = inputs
		net = slim.stack(net, slim.fully_connected, self.main_network_layers, activation_fn=tf.nn.relu, scope= name_scope, reuse=reuse,
			weights_initializer=self.initializer, biases_initializer=self.initializer)
		return net

	##
	## @brief      Sample a new head at each new episode (Bootstrapped DQN)
	##
	## @param      self  The object
	##
	def new_episode(self):
		self.head_used = np.random.randint(self.bootstrapping_config.nb_heads)

	##
	## @brief      Predict the Q Values for one neural network
	##
	## @param      self         The object
	## @param      name         The name of the neural network to use
	## @param      observation  The observation
	##
	## @return     the Q Values
	##
	def predict_q_values(self, name, observation):
		states = np.array(observation)
		action_values = self._tf_action_values[name][self.head_used].eval(
			feed_dict={self._tf_state[name]: states})
		return action_values

	##
	## @brief      Update the Neural Network using a Double Q Learning algorithm
	##
	## @param      self         The object
	## @param      prev_state   The previous state
	## @param      prev_action  The previous action
	## @param      reward       The reward
	## @param      next_state   The next state
	## @param      done         Is it a terminal state
	##
	def update_network(self, prev_state, prev_action, reward, next_state, done):
		if np.random.randint(0,2) == 0:
			#Update A
			self.create_experience('A', prev_state, prev_action, reward, next_state, done)
			self.train('A')
		else:
			#Update B
			self.create_experience('B', prev_state, prev_action, reward, next_state, done)
			self.train('B')

	##
	## @brief      Add an experience to the experience replay
	##
	## @param      self         The object
	## @param      name         The name
	## @param      prev_state   The previous state
	## @param      prev_action  The previous action
	## @param      reward       The reward
	## @param      next_state   The next state
	## @param      done         Is it a terminal state
	##
	def create_experience(self, name , prev_state, prev_action, reward, next_state, done):

		_prev_state = np.array(prev_state)
		_next_state = np.array(next_state)
		_reward = reward
		_done = done
		_action_mask = np.zeros(self._nb_actions)
		_action_mask[prev_action] = 1.0

		self.memory[name].store(ElementReplayMemory(_prev_state, _action_mask, _reward, _next_state, _done))

	##
	## @brief      Train a neural network using its own experience replay
	##
	## @param      self  The object
	## @param      name  The name of the neural network to train
	##
	def train(self, name):
		name_target = ''
		name_train = ''

		if name == 'A':
			name_train = 'A'
			name_target = 'B'
		elif name == 'B':
			name_train = 'B'
			name_target = 'A'		

		if self.memory[name].nbElementsStored() >= self.q_learning_config.size_replay_min_to_train:
			batch, w, e_ids = None, None, None
			if not self.replay_memory_config.use_prioritized_replay:
				batch = self.memory[name].generateRandomBatch(self._time)
				w = [1.0 for e in batch]
			else:
				batch, w, e_ids = self.memory[name].generateRandomBatch(self._time)

			y = []
			tab_q_values = self.predict_q_values(name_target,[e.next_state for e in batch])
			for i,experience in enumerate(batch):
				y_j = experience.reward
				if not experience.done:
					q_values = tab_q_values[i]
					y_j += self.q_learning_config.gamma * q_values.max()
				y.append(y_j)

			fatches = [self._tf_td_error_per_sample[name_train], self._tf_training[name_train]]

			feed = {
				self._tf_state[name_train]: [e.prev_state for e in batch],
				self._tf_action_mask[name_train]: [e.action_mask for e in batch],
				self._tf_y[name_train]: y,
				self._tf_w[name_train]: w,
				self._tf_learning_rate[name_train]: self.learning_rate
			}

			updated_td_errors, _ = self._tf_session.run(fatches, feed_dict=feed)

			if self.replay_memory_config.use_prioritized_replay:
				self.memory[name].update_priority(e_ids, updated_td_errors)

	##
	## @brief      Select the next action to perform
	##
	## @param      self         The object
	## @param      observation  The observation
	## @param      reward       The reward
	## @param      done         Is it a terminal state ?
	##
	## @return     The selected action
	##
	def act(self, observation, reward, done):
		self._time += 1
		self.update_learning_rate()
		q_values = (self.predict_q_values('A', [observation]) + self.predict_q_values('B', [observation])) / 2.0
		
		return self.policy.getAction(observation, self._action_space, q_values)