import csv
import numpy as np
import os


def load(filename):
	data = []
	with open(os.path.join('data', filename), "r") as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			for token in row:
				data.append(float(token))

	return np.array(data)


def split(data, perc = 0.7):
	data_size = len(data)
	train_size = int(data_size * perc)
	return (data[:train_size], data[train_size:])


def normalize(data):
	mean = np.mean(data)
	std = np.std(data)
	return (np.vectorize(lambda p: (p-mean)/std) (data), mean, std)


def sliding_window(trainset, window_size, stride = 1):
	windows = ((len(trainset) - (window_size + 1)) // stride) + 1
	matrix = np.array([trainset[(i*stride) : (i*stride)+window_size] for i in range(windows)])
	labels = np.array([trainset[(i*stride)+window_size] for i in range(windows)])
	return (matrix, labels)