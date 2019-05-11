import csv
import numpy as np
import os


def load(filename):
    data = []
    with open(os.path.join('data', filename), "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            for token in row:
                if token != "":
                    data.append(float(token))
                else:
                    data.append(None)

    return np.array(data, dtype=float)


def split(data, split):
    data_size = len(data)
    train_size = int(data_size * split)
    return (data[:train_size], data[train_size:])


""" mu=0 and stdev=1 normalization """
def normalize(data):
    data_clean = np.extract(np.logical_not(np.isnan(data)), data)
    mean = np.mean(data_clean)
    std = np.std(data_clean)

    def norm(val):
        if val is not None:
            return (val - mean) / std
        return None

    return (np.vectorize(norm)(data), mean, std)


def denormalize(data, mean, std):
    return np.vectorize(lambda p: (p * std) + mean)(data)


""" calc local derivative """
""" [1,3,-4,6] -> [1,2,-7,10] """
def differentiate(data, start=0):
    diff = []
    prev = start
    for point in data:
        diff.append(point - prev)
        prev = point

    return diff


def undifferentiate(data, start):
    undiff = []
    prev = start
    for point in data:
        prev = prev + point
        undiff.append(prev)

    return undiff
