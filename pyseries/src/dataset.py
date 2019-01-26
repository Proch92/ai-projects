import csv
import numpy as np


class Dataset():
	def __init__(self, filename):
		self.filename = filename
		self.data = list(self.load())
		(self.size, self.mean, self.std) = self.stats()
		self.normalize()


	def sliding_window(self, window_size, stride = 1):
		windows = int((self.size - (window_size + 1)) / stride) + 1
		matrix = np.array([self.data[(i*stride) : (i*stride)+window_size] for i in range(windows)])
		labels = np.array([self.data[(i*stride)+window_size] for i in range(windows)])

		return (matrix, labels)


	def normalize(self):
		self.data = [(point - self.mean) / self.std for point in self.data]


	def stats(self):
		size = len(self.data)
		mean = np.mean(self.data)
		std = np.std(self.data)

		return (size, mean, std)
	

	def load(self):
		with open(self.filename, "r") as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				for token in row:
					yield float(token)


	def toNumpy(self):
		return np.array(data)