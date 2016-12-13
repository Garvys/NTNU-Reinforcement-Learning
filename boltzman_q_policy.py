from policy import *
from copy import deepcopy

##
## @brief      Class for boltzamn Q policy.
##
class BoltzamnQPolicy(Policy):

	##
	## @brief      Constructs the object.
	##
	## @param      self  The object
	## @param      tau   The temperature
	##
	def __init__(self, tau = 1.0):
		self.prec_qvalues = None
		self.tau = tau

	##
	## @brief      Gets the action.
	##
	## @param      self          The object
	## @param      observation   The observation
	## @param      action_space  The action space
	## @param      qvalues       The qvalues
	##
	## @return     The action selected
	##
	def getAction(self, observation, action_space, qvalues):

		#All the qvalues are reduced to prevent the exponential from an overflow
		normalized_q_values = qvalues - min(qvalues[0])
		exp_q_values = np.exp(normalized_q_values / self.tau)
		probs = exp_q_values / np.sum(exp_q_values)
		probs = probs[0]

		probs = probs / sum(probs)
		
		a = np.random.choice(range(qvalues.shape[1]), p = probs)

		return a


		