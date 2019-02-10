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

	(eval_sequences, cuts_indexes) = split_evaluation_sequences(normalized)

	"""eval"""
	model = LSTM()

	fixed = np.empty(0)
	for head, tail in eval_sequences:
		head, start = differentiate(head)
		(_, projection) = model.evaluate(model_name, head, tail)
		head = undifferentiate(head, start)
		projection = undifferentiate(projection, head[-1])
		fixed = np.concatenate((fixed, head, projection))

	"""plot"""
	fixed_denorm = datautils.denormalize(fixed, mean, std)
	utils.plot_multiple([data, fixed_denorm], [0, 0], vertical_lines=cuts_indexes)


def split_evaluation_sequences(data):
	boolean_sequences = np.isnan(data)
	state = boolean_sequences[0]
	cuts_indexes = []
	cuts = []
	seq = []
	nan_counter = 0
	for i, nan in enumerate(boolean_sequences):
		# state change
		if nan != state:
			cuts_indexes.append(i)
			if state == True:
				print("cut: index {}, length {}".format(i, nan_counter))
				cuts.append((seq, nan_counter))
				seq = []
				nan_counter = 0

		if nan:
			nan_counter = nan_counter + 1
		else:
			seq.append(data[i])

		state = nan

	return (cuts, cuts_indexes)


if __name__ == '__main__':
	main()