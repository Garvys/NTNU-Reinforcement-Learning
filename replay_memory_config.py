##
## @brief      Class for specifying the configuration of the Experience Replay
##
class ReplayMemoryConfig(object):

	##
	## @brief      Constructs the object.
	##
	## @param      self                    The object
	## @param      use_prioritized_replay  The use prioritized replay
	## @param      memorySize              The memory size
	## @param      alpha                   The alpha
	## @param      beta_zero               The beta zero
	## @param      total_steps             The total steps
	##
	def __init__(self, use_prioritized_replay = False, memorySize = 500, alpha = 0.7,
		beta_zero = 0.5, total_steps = 4000):
		self.use_prioritized_replay = use_prioritized_replay
		self.memorySize = memorySize

		#Prioritized Replay
		self.alpha = alpha
		self.beta_zero = beta_zero
		self.total_steps = total_steps

	##
	## @brief      Returns a string representation of the object.
	##
	## @param      self  The object
	##
	## @return     String representation of the object.
	##
	def __repr__(self):
		return 'Replay config : {} {} {} {} {}'.format(self.use_prioritized_replay, self.memorySize, self.alpha, self.beta_zero, self.total_steps)
