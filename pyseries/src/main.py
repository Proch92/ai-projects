from lstm import LSTM
import datautils
import utils
import sys

def main():
	argv = sys.argv

	if len(argv) != 2:
		print('Usage: ' + argv[0] + ' dataset')
		sys.exit(0)

	data = datautils.load(argv[1])
	normalized, mean, std = datautils.normalize(data)
	(train, validation, test) = datautils.split(normalized, 0.6, 0.2, 0.2)
	
	utils.plot_data(data)

	print("training set length: {}".format(len(train)))
	print("test set length: {}".format(len(test)))

	"""train"""
	model = LSTM()
	time_steps = 20 # window size
	batch_size = 1
	model.train(train, validation, 500, batch_size, time_steps)

	"""test"""
	head = int(len(test) * 0.6)
	tail = len(test) - head
	(guided, projection) = model.evaluate(test[:head], tail)
	
	"""plot"""
	testset_denorm = datautils.denormalize(test, mean, std)
	results_denorm = datautils.denormalize(projection, mean, std)
	utils.plot_multiple([testset_denorm, guided, results_denorm], [0, 1, head+1])

if __name__ == '__main__':
	main()