import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history, validation=False):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Square Error [$MPG^2$]')
	plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
	if validation:
		plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Validation Error')
	plt.legend()
	plt.ylim([0,5])
	plt.show()


def plot_data(sequence):
	plt.figure()
	plt.xlabel('Time')
	plt.ylabel('Value')
	plt.plot(range(len(sequence)), sequence)
	plt.ylim([min(sequence), max(sequence)])
	plt.show()


def plot_multiple(sequences, shift = None):
	if not shift:
		shift = [0] * len(sequences)

	max_y = max([max(seq) for seq in sequences])
	min_y = min([min(seq) for seq in sequences])

	plt.figure()
	plt.xlabel('Time')
	plt.ylabel('Value')
	for i, sequence in enumerate(sequences):
		plt.plot([t + shift[i] for t in range(len(sequence))], sequence)
	plt.ylim([min_y, max_y])
	plt.show()
