##
## @brief      Class for specifying the configuration of the dueling architecture
##
class DuelingConfig:

	##
	## @brief      Constructs the object.
	##
	## @param      self            The object
	## @param      use_dueling     Is the dueling architecture used
	## @param      size_net_value  The number of neurons in the layer representing the state-value function
	## @param      size_net_adv    The number if neurons in the layer representing the advantage-value function
	##
	def __init__(self, use_dueling = False, size_net_value = 100, size_net_adv = 100):
		self.use_dueling = use_dueling
		self.size_net_value = size_net_value
		self.size_net_adv = size_net_adv

	##
	## @brief      Returns a string representation of the object.
	##
	## @param      self  The object
	##
	## @return     String representation of the object.
	##
	def __repr__(self):
		return "Dueling config : {} {} {}".format(self.use_dueling, self.size_net_value, self.size_net_adv)