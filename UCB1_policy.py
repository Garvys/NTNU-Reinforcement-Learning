from policy import *
from copy import deepcopy
import math

##
## @brief      Class implementing the UCB1 policy : Optimism in face of Uncertainty
##
class UCB1Policy(Policy):

	##
	## @brief      Constructs the object.
	##
	## @param      self        The object
	## @param      nb_actions  The number of actions
	##
	def __init__(self, nb_actions):
		self._time = 1
		self._count_actions_selected = np.ones(nb_actions)

	##
	## @brief      Select the action to perform
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

		for i in range(qvalues.shape[1]):
			qvalues[0,i] += math.sqrt(2.0*np.log(self._time) / self._count_actions_selected[i])
		
		action = EpsilonGreedyPolicy(0).getAction(observation, action_space, qvalues)

		self._count_actions_selected[action] += 1

		return action
		