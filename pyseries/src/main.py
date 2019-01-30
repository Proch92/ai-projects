#from fnn import Fnn
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
	trainset, testset = datautils.split(normalized, 0.7)
#	utils.plot_data(trainset)
	
#	model = Fnn(window_size=10)
#	model.train(trainset, epochs=200)

	print("training set length: {}".format(len(trainset)))
	print("test set length: {}".format(len(testset)))

	model = LSTM()
	time_steps = 10
	batch_size = 5
	model.train(trainset, 50, batch_size, time_steps)

	head = int(len(testset) * 0.6)
	tail = len(testset) - head
	results = model.evaluate(testset[:head], tail)
	
	testset_denorm = datautils.denormalize(testset, mean, std)
	results_denorm = datautils.denormalize(results, mean, std)
	utils.plot_multiple([testset_denorm, results_denorm], [0, head + time_steps])

if __name__ == '__main__':
	main()