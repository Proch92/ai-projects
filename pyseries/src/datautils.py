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


def denormalize(data, mean, std):
	return np.vectorize(lambda p: (p * std) + mean) (data)


def streamline(data):
	if len(data) == 0:
		return []

	if len(data) == 1:
		return data[0]

	results = data[0].tolist()
	data = data[1:]

	for vec in data:
		results.append(vec[-1])

	return results