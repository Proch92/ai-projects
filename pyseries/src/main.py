from fnn import Fnn
from rnn import Rnn
import dataset
import utils
import sys

def main():
	argv = sys.argv

	if len(argv) != 2:
		print('Usage: ' + argv[0] + ' dataset')
		sys.exit(0)

	data = dataset.load(argv[1])
	normalized, mean, std = dataset.normalize(data)
	trainset, testset = dataset.split(normalized, 0.7)
#	utils.plot_data(trainset)
	
#	model = Fnn(window_size=10)
#	model.train(trainset, epochs=200)

	model = Rnn(time_steps=10)
	model.train(trainset, epochs=10, batch_size=10)

if __name__ == '__main__':
	main()