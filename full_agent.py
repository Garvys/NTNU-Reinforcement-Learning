import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from copy import deepcopy
from policy import *
from replay_memory import *
from prioritized_replay_memory import *

class DuelingConfig:
	"""docstring for DuelingConfig"""
	def __init__(self, use_dueling = False, size_net_value = 100, size_net_adv = 100):
		self.use_dueling = use_dueling
		self.size_net_value = size_net_value
		self.size_net_adv = size_net_adv

	def __repr__(self):
		return "Dueling config : {} {} {}".format(self.use_dueling, self.size_net_value, self.size_net_adv)

class BootstrappingConfig:
	"""docstring for BootstrappingConfig"""
	def __init__(self, nb_heads = 1, size_layer_head = 100):
		self.nb_heads = nb_heads
		self.size_layer_head = size_layer_head

	def __repr__(self):
		return "Bootstrapping config : {} {}".format(self.nb_heads, self.size_layer_head)

class QLearningConfig:
	"""docstring for QLearningConfig"""
	def __init__(self, batch_size = 200, gamma = 0.95, size_replay_min_to_train = 100,
		learning_rate_start = 1e-3, learning_rate_end = 1e-8, time_learning_rate_end = 10000):
		self.batch_size = batch_size
		self.gamma = gamma
		self.size_replay_min_to_train = size_replay_min_to_train

		#Learning Rate
		self.learning_rate_start = learning_rate_start
		self.learning_rate_end = learning_rate_end
		self.time_learning_rate_end = time_learning_rate_end

	def __repr__(self):
		return "QLearning condig : {} {} {} {} {} {}".format(self.batch_size, self.gamma, self.size_replay_min_to_train, self.learning_rate_start, self.learning_rate_end, self.time_learning_rate_end)

class ReplayMemoryConfig(object):
	"""docstring for ReplayMemoryConfig"""
	def __init__(self, use_prioritized_replay = False, memorySize = 500, alpha = 0.7,
		beta_zero = 0.5, total_steps = 4000):
		self.use_prioritized_replay = use_prioritized_replay
		self.memorySize = memorySize

		#Prioritized Replay
		self.alpha = alpha
		self.beta_zero = beta_zero
		self.total_steps = total_steps

	def __repr__(self):
		return 'Replay config : {} {} {} {} {}'.format(self.use_prioritized_replay, self.memorySize, self.alpha, self.beta_zero, self.total_steps)
		
		
		


class DoubleDQNAgent:
	"""docstring for DQNAgent"""
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

	def update_networks(self):
		self._tf_session.run(self.copyTargetQNetworkOperations)

	def update_learning_rate(self):

		a = (self.q_learning_config.learning_rate_start - self.q_learning_config.learning_rate_end) / (0 - self.q_learning_config.time_learning_rate_end)
		b = self.q_learning_config.learning_rate_start

		lr = a*self._time + b
		self.learning_rate = max(lr, self.q_learning_config.learning_rate_end)

	def build_q_network(self, name):
		#Create a network

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

			#loss = tf.reduce_mean(td_error)

			#tab_loss.append(loss)
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

	def create_network(self, inputs, name_scope, reuse = False):
		net = inputs
		net = slim.stack(net, slim.fully_connected, self.main_network_layers, activation_fn=tf.nn.relu, scope= name_scope, reuse=reuse,
			weights_initializer=self.initializer, biases_initializer=self.initializer)
		return net

	def new_episode(self):
		self.head_used = np.random.randint(self.bootstrapping_config.nb_heads)

	def predict_q_values(self, name, observation):
		states = np.array(observation)
		action_values = self._tf_action_values[name][self.head_used].eval(
			feed_dict={self._tf_state[name]: states})
		return action_values

	def update_network(self, prev_state, prev_action, reward, next_state, done):
		if np.random.randint(0,2) == 0:
			#Update A
			self.create_experience('A', prev_state, prev_action, reward, next_state, done)
			self.train('A')
		else:
			#Update B
			self.create_experience('B', prev_state, prev_action, reward, next_state, done)
			self.train('B')

	def create_experience(self, name , prev_state, prev_action, reward, next_state, done):
		"""
		keep an experience for later training.
		"""
		_prev_state = np.array(prev_state)
		_next_state = np.array(next_state)
		_reward = reward
		_done = done
		_action_mask = np.zeros(self._nb_actions)
		_action_mask[prev_action] = 1.0

		self.memory[name].store(ElementReplayMemory(_prev_state, _action_mask, _reward, _next_state, _done))

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
			#print(np.shape(w))
			#print(w)
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

	def act(self, observation, reward, done):
		self._time += 1
		self.update_learning_rate()
		q_values = (self.predict_q_values('A', [observation]) + self.predict_q_values('B', [observation])) / 2.0
		
		return self.policy.getAction(observation, reward, done, self._action_space, q_values)