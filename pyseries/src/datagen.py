import sys
import os
import math

DATASET_SIZE = 1000


def sin(i):
	START = 0.0
	STEPS = 0.01
	return math.sin(START + (i * STEPS))


def linear(i):
	START = 0.0
	STEPS = 0.01
	return START + (STEPS * i)


def gen(func):
	with open(os.path.join('data', func + '.csv'), "w") as f:
		for i in range(DATASET_SIZE):
			f.write(str(eval(func)(i)) + ',')
		f.seek(f.tell() - 1, os.SEEK_SET)
		f.truncate()


if __name__ == '__main__':
	argv = sys.argv

	if len(argv) != 2:
		print('Usage: datagen.py function_name')
		exit(0)

	func = argv[1]

	gen(func)
