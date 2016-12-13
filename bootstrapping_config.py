##
## @brief      Class for specifying the configuration of the bootstrapping
##
class BootstrappingConfig:

	##
	## @brief      Constructs the object.
	##
	## @param      self             The object
	## @param      nb_heads         The number of heads
	## @param      size_layer_head  The number of neurons in the hidden layer inside each head
	##
	def __init__(self, nb_heads = 1, size_layer_head = 100):
		self.nb_heads = nb_heads
		self.size_layer_head = size_layer_head

	##
	## @brief      Returns a string representation of the object.
	##
	## @param      self  The object
	##
	## @return     String representation of the object.
	##
	def __repr__(self):
		return "Bootstrapping config : {} {}".format(self.nb_heads, self.size_layer_head)