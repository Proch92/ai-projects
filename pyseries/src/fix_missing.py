from lstm import LSTM
import datautils
import utils
import sys
import numpy as np


def main():
	argv = sys.argv

	if len(argv) != 3:
		print('Usage: ' + argv[0] + ' model_name dataset')
		sys.exit(0)

	model_name = argv[1]
	data = datautils.load(argv[2])
	normalized, mean, std = datautils.normalize(data)

	eval_sequences = split_evaluation_sequences(normalized)

	"""eval"""
	model = LSTM()

	fixed = np.empty(0)
	for data, tail in eval_sequences:
		(_, projection) = model.evaluate(model_name, data, tail)
		fixed = np.concatenate((fixed, data, projection))

	"""plot"""
	fixed_denorm = datautils.denormalize(fixed, mean, std)
	utils.plot_multiple([fixed_denorm], [0])


def split_evaluation_sequences(data):
	boolean_sequences = np.isnan(data)
	state = boolean_sequences[0]
	seq = []
	nan_counter = 0
	for i, nan in enumerate(boolean_sequences):
		# state change
		if nan != state and state == True:
			yield (seq, nan_counter)
			seq = []
			nan_counter = 0

		if nan:
			nan_counter = nan_counter + 1
		else:
			seq.append(data[i])

		state = nan



if __name__ == '__main__':
	main()