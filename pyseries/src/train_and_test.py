from lstm import LSTM
import datautils
import utils
import sys

def main():
	argv = sys.argv

	if len(argv) != 2:
		print('Usage: ' + argv[0] + ' model_name dataset')
		sys.exit(0)

	model_name = argv[1]
	data = datautils.load(argv[2])
	normalized, mean, std = datautils.normalize(data)
	(train, test) = datautils.split(normalized, 0.7)
	
	# utils.plot_data(data)

	print("training set length: {}".format(len(train)))
	print("test set length: {}".format(len(test)))

	"""train"""
	model = LSTM()
	time_steps = 20 # window size
	batch_size = 5 # data augmentation
	model.train(model_name, train, 130, batch_size, time_steps)

	"""test"""
	head = int(len(test) * 0.6)
	tail = len(test) - head
	(guided, projection) = model.evaluate(model_name, test[:head], tail)
	
	"""plot"""
	testset_denorm = datautils.denormalize(test, mean, std)
	guided_denorm = datautils.denormalize(guided, mean, std)
	results_denorm = datautils.denormalize(projection, mean, std)
	utils.plot_multiple([testset_denorm, guided_denorm, results_denorm], [0, 1, head+1])

if __name__ == '__main__':
	main()