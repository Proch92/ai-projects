import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Square Error [$MPG^2$]')
	plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
	plt.legend()
	plt.ylim([0,5])
	plt.show()


def plot_data(dataset):
	plt.figure()
	plt.xlabel('Time')
	plt.ylabel('Value')
	plt.plot(range(dataset.size), dataset.get_denormalized_data())
	plt.ylim([dataset.min, dataset.max])
	plt.show()