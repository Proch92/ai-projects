from lstm import LSTM
import datautils
import utils
import sys

def main():
	argv = sys.argv

	if len(argv) != 3:
		print('Usage: ' + argv[0] + ' model_name dataset')
		sys.exit(0)

	model_name = argv[1]
	data = datautils.load(argv[2])
	normalized, mean, std = datautils.normalize(data)
	normalized, start = datautils.differentiate(normalized)
	(train, test) = datautils.split(normalized, 0.7)
	
	# utils.plot_data(data)

	print("training set length: {}".format(len(train)))
	print("test set length: {}".format(len(test)))

	"""train"""
	model = LSTM()
	time_steps = 20 # window size
	batch_size = 5 # data augmentation
	history = model.train(model_name, train, 40, batch_size, time_steps)
	utils.plot_history(history)

	"""test"""
	head = int(len(test) * 0.6)
	tail = len(test) - head
	(guided, projection) = model.evaluate(model_name, test[:head], tail)
	
	"""plot"""
	start_test = start + sum(train)
	start_projection = start_test + sum(test)
	test = datautils.undifferentiate(test, start_test)
	projection = datautils.undifferentiate(projection, start_projection)
	testset_denorm = datautils.denormalize(test, mean, std)
	results_denorm = datautils.denormalize(projection, mean, std)
	utils.plot_multiple([testset_denorm, results_denorm], [0, head+1])

if __name__ == '__main__':
	main()