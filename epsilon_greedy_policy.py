from policy import *

##
## @brief      Class for the epsilon greedy policy.
##
class EpsilonGreedyPolicy(Policy):

	##
	## @brief      Constructs the object.
	##
	## @param      self                The object
	## @param      epsilon             The epsilon
	## @param      decay               The decay
	## @param      epsilon_decay_time  The epsilon decay time = the number of time steps between each decay
	## @param      epsilon_decay_rate  The epsilon decay rate = how much the epsilon should be reduced
	##
	def __init__(self, epsilon, decay = False, epsilon_decay_time = 500, epsilon_decay_rate = 0.95):
		self._epsilon = epsilon
		self._decay = decay
		self._epsilon_decay_time = epsilon_decay_time
		self._epsilon_decay_rate = epsilon_decay_rate
		self._time = 0

	##
	## @brief      Select the next action
	##
	## @param      self          The object
	## @param      observation   The observation
	## @param      action_space  The action space
	## @param      qvalues       The qvalues
	##
	## @return     The action selected
	##
	def getAction(self, observation, action_space, qvalues):
		self._time += 1
		if self._decay and self._time % self._epsilon_decay_time == 0:
			self._epsilon *= self._epsilon_decay_rate
			if self._epsilon < 0.001:
				self._epsilon = 0.0

		action = None
		if np.random.rand() >= self._epsilon:
			action = np.argmax(qvalues)
		else:
			action = action_space.sample()

		return action


		