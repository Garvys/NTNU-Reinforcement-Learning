from copy import deepcopy
import numpy as np

##
## @brief      Class for every tuple that is stored in the reply memory.
##
class ElementReplayMemory:

	##
	## @brief      Constructs the object.
	##
	## @param      self         The object
	## @param      prev_state   The previous state
	## @param      action_mask  The action mask
	## @param      reward       The reward
	## @param      next_state   The next state
	## @param      done         If this is a terminal state
	##
	def __init__(self, prev_state, action_mask, reward, next_state, done):
		self.prev_state = prev_state
		self.action_mask = action_mask
		self.reward = reward
		self.next_state = next_state
		self.done = done


##
## @brief      Class for the experience replay memory.
##
class ReplayMemory:

	##
	## @brief      Constructs the object.
	##
	## @param      self       The object
	## @param      sizeMax    How many experience will be stored in the memory
	## @param      sizeBatch  How many experience are sampled to update the netword (size of the mini batch)
	##
	def __init__(self, sizeMax, sizeBatch):
		self.sizeMax = sizeMax
		self.sizeBatch = sizeBatch
		self.memory = np.array([None for i in range(sizeMax)])

		self.full = False
		self.next_idx = 0

	##
	## @brief      Store an element in the experience replay
	##
	## @param      self     The object
	## @param      element  the element to store
	##
	def store(self, element):
		if self.full:
			self.memory[np.random.randint(0,self.sizeMax)] = deepcopy(element)
		else:
			self.memory[self.next_idx] = deepcopy(element)
			self.next_idx += 1
			if self.next_idx >= self.sizeMax:
				self.full = True

	##
	## @brief      Determines if the experience replay memory is full.
	##
	## @param      self  The object
	##
	## @return     True if the memory is full, False otherwise.
	##
	def isFull(self):
		return self.full

	##
	## @brief      Determines the number of elements stored in the replay memory
	##
	## @param      self  The object
	##
	## @return     the number of elements really stored in the replay memory
	##
	def nbElementsStored(self):
		if self.full:
			return self.sizeMax
		else:
			return self.next_idx

	##
	## @brief      Sample uniformally a mini-batch from the replay
	##
	## @param      self         The object
	## @param      global_step  The global step
	##
	## @return     the mini batch
	##
	def generateRandomBatch(self, global_step):
		borneMax = self.sizeMax
		if not self.full:
			borneMax = self.next_idx
		ixs = np.random.choice(borneMax, self.sizeBatch, replace=True)
		return self.memory[ixs]





		