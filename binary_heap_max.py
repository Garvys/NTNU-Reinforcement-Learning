from copy import deepcopy


class ElementBinaryHeap:
	"""docstring for ElementBinaryHeap"""
	def __init__(self, data, priority):
		self.priority = priority
		self.data = data
		

class BinaryHeapMax(object):
	"""docstring for BinaryHeapMax"""
	def __init__(self):
		self.tab = []
		self.sizeCour = 0
		self.element_to_index = {}

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

	def is_root(self, index):
		return index == 0

	def get_index_father(self, index):
		return int((index - 1) / 2)

	def get_indexes_sons(self, index):
		return [2*index + 1, 2*index + 2]

	def has_sons(self, index):
		return (2*index + 1) < len(self.tab)

	def swap_nodes(self, index_1, index_2):
		elt1 = deepcopy(self.tab[index_1])
		elt2 = deepcopy(self.tab[index_2])

		self.tab[index_1] = elt2
		self.tab[index_2] = elt1

		self.element_to_index[elt2.data] = index_1
		self.element_to_index[elt1.data] = index_2

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


	def remove(self, element_to_remove):
		self.tab[element_to_remove] = deepcopy(self.tab[-1])
		self.element_to_index[self.tab[-1].data] = element_to_remove
		del self.element_to_index[self.tab[-1].data]
		self.tab.pop()

		self.percolate_down(element_to_remove)

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

	def size(self):
		return self.tab.size()

	def get_max_priority(self):
		if len(self.tab) > 0:
			return self.tab[0].priority
		else:
			return 1.0

	def print_heap(self):
		#### Generate code in DOT Language
		print('digraph G {')

		for i in range(1, len(self.tab)):
			print('{} -> {}'.format(self.tab[self.get_index_father(i)].priority, self.tab[i].priority))

		print('}')

	def sort_heap(self):
		self.tab.sort(key = lambda x: x.priority, reverse = True)

		for i in range(len(self.tab)):
			element = self.tab[i]
			self.element_to_index[element.data] = i

	def update(self, experience_id, new_priority):
		#print(experience_id, new_priority, self.tab)
		index_cour = self.element_to_index[experience_id]
		self.tab[index_cour].priority = new_priority

		self.percolate_down(index_cour)
		self.percolate_up(index_cour)













		