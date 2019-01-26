from fnn import Fnn
from rnn import Rnn
from dataset import Dataset
import utils
import sys

def main():
	argv = sys.argv

	if len(argv) != 2:
		print('Usage: ' + argv[0] + ' dataset')

	dataset = dataset.load(argv[1])
	dataset, mean, std = dataset.normalize(dataset)
	trainset, testset = dataset.split(0.7)
#	utils.plot_data(dataset)
	
	model = Fnn(window_size=10)
	model.train(trainset, epochs=200)

	model = Rnn(time_steps=10)
	model.train(trainset, epochs=200)

if __name__ == '__main__':
	main()