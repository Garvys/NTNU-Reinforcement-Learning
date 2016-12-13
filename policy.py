import sys
import numpy as np


##
## @brief      Basic class for every policy
##
class Policy:

	##
	## @brief      Select the next action to perform
	##
	## @param      self          The object
	## @param      observation   Current state of the environment
	## @param      action_space  The action space to be able to sample an action uniformally
	## @param      qvalues       The qvalues
	##
	## @return     The action selected
	##
	def getAction(self, observation, action_space, qvalues):
		print("Policy not implemented", file=sys.stderr)
		exit(1)
		

from epsilon_greedy_policy import *
from boltzman_q_policy import *
from UCB1_policy import *