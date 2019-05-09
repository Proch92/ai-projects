from lstm import LSTM
import datautils
import utils
import sys

def main():
	argv = sys.argv

	if len(argv) != 4:
		print('Usage: ' + argv[0] + ' model_name dataset projection_length')
		sys.exit(0)

	model_name = argv[1]
	data = datautils.load(argv[2])
	tail = int(argv[3])
	normalized, mean, std = datautils.normalize(data)
	diff = datautils.differentiate(normalized)

	"""eval"""
	model = LSTM()
	projection = model.evaluate(model_name, diff, tail)
	
	"""plot"""
	results_undiff = datautils.undifferentiate(projection, normalized[-1])
	results_denorm = datautils.denormalize(results_undiff, mean, std)
	utils.plot_multiple([data, results_denorm], [0, len(data)])

if __name__ == '__main__':
	main()