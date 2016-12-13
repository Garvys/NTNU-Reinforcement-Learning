from copy import deepcopy


##
## @brief      Class for each element stored in the binary Heap : a priority and a data
##
class ElementBinaryHeap:

	##
	## @brief      Constructs the object.
	##
	## @param      self      The object
	## @param      data      The data
	## @param      priority  The priority
	##
	def __init__(self, data, priority):
		self.priority = priority
		self.data = data
		

##
## @brief      Implementation of a max-binary heap
##
class BinaryHeapMax(object):

	##
	## @brief      Constructs the object.
	##
	## @param      self  The object
	##
	def __init__(self):
		self.tab = []
		self.sizeCour = 0
		self.element_to_index = {}

	##
	## @brief      Insert a new element in the heap
	##
	## @param      self               The object
	## @param      element_to_insert  The element to insert
	## @param      priority           The priority
	##
	def insert(self, element_to_insert, priority):

		#Add the element at the end of the priority queue
		if element_to_insert in self.element_to_index:
			index = self.element_to_index[element_to_insert]
			self.tab[index].priority = priority

			self.percolate_up(index)
			self.percolate_down(index)
		else:
			self.tab.append(ElementBinaryHeap(element_to_insert, priority))
			self.element_to_index[element_to_insert] = len(self.tab) - 1

			self.percolate_up(len(self.tab) - 1)

	##
	## @brief      Determines if the element is the the top priority.
	##
	## @param      self   The object
	## @param      index  The index
	##
	## @return     True if it is the root, False otherwise.
	##
	def is_root(self, index):
		return index == 0

	##
	## @brief      Gets the index of the father.
	##
	## @param      self   The object
	## @param      index  The index
	##
	## @return     The index of the father.
	##
	def get_index_father(self, index):
		return int((index - 1) / 2)

	##
	## @brief      Gets the indexes of the two sons.
	##
	## @param      self   The object
	## @param      index  The index
	##
	## @return     The indexes of the sons .
	##
	def get_indexes_sons(self, index):
		return [2*index + 1, 2*index + 2]

	##
	## @brief      Determines if it has sons.
	##
	## @param      self   The object
	## @param      index  The index
	##
	## @return     True if has sons, False otherwise.
	##
	def has_sons(self, index):
		return (2*index + 1) < len(self.tab)

	##
	## @brief      Swap two nodes in the heap
	##
	## @param      self     The object
	## @param      index_1  The index 1
	## @param      index_2  The index 2
	##
	def swap_nodes(self, index_1, index_2):
		elt1 = deepcopy(self.tab[index_1])
		elt2 = deepcopy(self.tab[index_2])

		self.tab[index_1] = elt2
		self.tab[index_2] = elt1

		self.element_to_index[elt2.data] = index_1
		self.element_to_index[elt1.data] = index_2

	##
	## @brief      Percolate up the node to place it at the right place
	##
	## @param      self           The object
	## @param      index_element  The index element
	##
	def percolate_up(self, index_element):

		index_cour = deepcopy(index_element)

		while True:
			if self.is_root(index_cour):
				break

			index_father = self.get_index_father(index_cour)

			if self.tab[index_father].priority < self.tab[index_cour].priority:
				self.swap_nodes(index_father, index_cour)
				index_cour = index_father
			else:
				break


	##
	## @brief      Remove a node from the heap
	##
	## @param      self               The object
	## @param      element_to_remove  The element to remove
	##
	def remove(self, element_to_remove):
		self.tab[element_to_remove] = deepcopy(self.tab[-1])
		self.element_to_index[self.tab[-1].data] = element_to_remove
		del self.element_to_index[self.tab[-1].data]
		self.tab.pop()

		self.percolate_down(element_to_remove)

	##
	## @brief      Percolate down the node to put it at the right place
	##
	## @param      self           The object
	## @param      index_element  The index element
	##
	def percolate_down(self,index_element):

		index_cour = deepcopy(index_element)

		while True:

			if not self.has_sons(index_cour):
				break

			l = []

			for index_son in self.get_indexes_sons(index_cour):
				if index_son < len(self.tab) :
					l.append([self.tab[index_son].priority, index_son])

			greater_priority, index_greater_son = max(l)

			if greater_priority > self.tab[index_cour].priority:
				self.swap_nodes(index_cour, index_greater_son)
				index_cour = index_greater_son
			else:
				break

	##
	## @brief      Number of elements in the heap
	##
	## @param      self  The object
	##
	## @return     Size of the heap
	##
	def size(self):
		return self.tab.size()

	##
	## @brief      Gets the maximum priority.
	##
	## @param      self  The object
	##
	## @return     The maximum priority.
	##
	def get_max_priority(self):
		if len(self.tab) > 0:
			return self.tab[0].priority
		else:
			return 1.0

	##
	## @brief      Print on stdout a represenation of the heap in DOT Language
	##
	## @param      self  The object
	##
	def print_heap(self):
		print('digraph G {')

		for i in range(1, len(self.tab)):
			print('{} -> {}'.format(self.tab[self.get_index_father(i)].priority, self.tab[i].priority))

		print('}')

	##
	## @brief      Sort the heap accorsding to the priority
	##
	## @param      self  The object
	##
	def sort_heap(self):
		self.tab.sort(key = lambda x: x.priority, reverse = True)

		for i in range(len(self.tab)):
			element = self.tab[i]
			self.element_to_index[element.data] = i

	##
	## @brief      Update the priority of an element in the heap and places it correctly
	##
	## @param      self           The object
	## @param      experience_id  The experience identifier
	## @param      new_priority   The new priority
	##
	def update(self, experience_id, new_priority):
		#print(experience_id, new_priority, self.tab)
		index_cour = self.element_to_index[experience_id]
		self.tab[index_cour].priority = new_priority

		self.percolate_down(index_cour)
		self.percolate_up(index_cour)













		