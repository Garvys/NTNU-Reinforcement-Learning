from replay_memory import *		
from binary_heap_max import *
import numpy as np

##
## @brief      Class for the Prioritized Experiecne Replay Memory.
##
class PrioritizedReplayMemory:

	##
	## @brief      Constructs the object.
	##
	## @param      self                  The object
	## @param      sizeMax               The size maximum
	## @param      replay_memory_config  The replay memory configuration
	## @param      batch_size            The batch size
	## @param      learn_start           The learning start
	##
	def __init__(self, sizeMax, replay_memory_config , batch_size = 500, learn_start = 100):

		#Implemented using a binary heap
		self.priority_queue = BinaryHeapMax()
		self.sizeMax = sizeMax
		self.batch_size = batch_size

		self.alpha = replay_memory_config.alpha
		self.beta_zero = replay_memory_config.beta_zero
		self.nb_partitions = batch_size
		self._experience = [None for e in range(sizeMax)]
		self.next_experience_index = 0
		self.total_steps = replay_memory_config.total_steps
		self.learn_start = learn_start

		self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

		self.probas = []

		for i in range(sizeMax):
			rank = i + 1
			p_i = pow(1.0 / float(rank), self.alpha)
			self.probas.append(p_i)

	##
	## @brief      Gets the number of elements stored in the memory
	##
	## @param      self  The object
	##
	## @return     The number of elements stored in the memory
	##
	def nbElementsStored(self):
		return len(self.priority_queue.tab)

	##
	## @brief      Store a new element in the memory
	##
	## @param      self        The object
	## @param      experience  The experience to store
	##
	def store(self, experience):
		insert_index = self.next_experience_index % self.sizeMax
		self._experience[insert_index] = deepcopy(experience)

		self.priority_queue.insert(insert_index, self.priority_queue.get_max_priority())

		self.next_experience_index += 1


	##
	## @brief      Update the priority of some experiences stored in the memory
	##
	## @param      self                  The object
	## @param      expericences_indices  The expericences indices that have to be updated
	## @param      deltas                The new delatas
	##
	def update_priority(self, expericences_indices, deltas):
		for i in range(len(expericences_indices)):
			#print(deltas[i], abs(deltas[i]))
			self.priority_queue.update(expericences_indices[i], abs(deltas[i]))

	##
	## @brief      Generate the boundaries and the probabilities of each segment.
	##     			Allow to sample more efficiently the mini batch from the memory
	##
	## @param      self  The object
	##
	## @return     The boundaries and the probabilities of each segment
	##
	def get_boundaries(self):
		boundaries = []

		nb_elements_stored = len(self.priority_queue.tab)

		partition_size = nb_elements_stored / self.nb_partitions

		probas_segments = []

		if nb_elements_stored < self.nb_partitions:
			for i in range(0, nb_elements_stored):
				boundaries.append(i)
				probas_segments.append(self.probas[i])
		else:
			for i in range(self.nb_partitions):
				index = round(i*partition_size)
				boundaries.append(index)
				probas_segments.append(self.probas[i])

		boundaries.append(nb_elements_stored)
		sum_probas = sum(probas_segments)
		probas_segments = [p / sum_probas for p in probas_segments]

		return boundaries, probas_segments

	##
	## @brief      Sample a mini batch from the prioritized relay memory
	##
	## @param      self         The object
	## @param      global_step  The global step
	##
	## @return     mini-batch
	##
	def generateRandomBatch(self, global_step):

		beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
		#beta = self.beta_zero
		self.priority_queue.sort_heap()
		nb_elements_stored = len(self.priority_queue.tab)

		boundaries, probas_segments = self.get_boundaries()
		experience_ids = []
		experience_batch = []
		list_indexes = []
		probas_experiences_selected = []
		for i in range(self.batch_size):
			index_segment = np.random.choice(list(range(0, len(boundaries)-1)), p = probas_segments)
			index = np.random.randint(boundaries[index_segment], boundaries[index_segment+1])
			e_id = self.priority_queue.tab[index].data

			list_indexes.append(index)
			experience_ids.append(e_id)
			experience_batch.append(self._experience[e_id])
			probas_experiences_selected.append(probas_segments[index_segment])


		w = [pow(nb_elements_stored * P, - beta) for P in probas_experiences_selected]
		w_max = max(w)
		w = [e / w_max for e in w]

		return experience_batch, w, experience_ids
