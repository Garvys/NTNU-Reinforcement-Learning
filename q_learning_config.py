##
## @brief      Class for specifying the hyper parameters of the Q Learning algorithm
##
class QLearningConfig:

	##
	## @brief      Constructs the object.
	##
	## @param      self                      The object
	## @param      batch_size                The batch size
	## @param      gamma                     The gamma (= the horizon)
	## @param      size_replay_min_to_train  The size replay minimum to train
	## @param      learning_rate_start       The learning rate start
	## @param      learning_rate_end         The learning rate end
	## @param      time_learning_rate_end    How many time steps needed to completely anneal the learning rate
	##
	def __init__(self, batch_size = 200, gamma = 0.95, size_replay_min_to_train = 100,
		learning_rate_start = 1e-3, learning_rate_end = 1e-8, time_learning_rate_end = 10000):
		self.batch_size = batch_size
		self.gamma = gamma
		self.size_replay_min_to_train = size_replay_min_to_train

		#Learning Rate
		self.learning_rate_start = learning_rate_start
		self.learning_rate_end = learning_rate_end
		self.time_learning_rate_end = time_learning_rate_end

	##
	## @brief      Returns a string representation of the object.
	##
	## @param      self  The object
	##
	## @return     String representation of the object.
	##
	def __repr__(self):
		return "QLearning condig : {} {} {} {} {} {}".format(self.batch_size, self.gamma, self.size_replay_min_to_train, self.learning_rate_start, self.learning_rate_end, self.time_learning_rate_end)
