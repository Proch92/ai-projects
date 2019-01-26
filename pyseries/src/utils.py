import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history):
		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch

		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Mean Abs Error [MPG]')
		plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
		plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
		plt.legend()
		plt.ylim([0,1])

		plt.show()

		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Mean Square Error [$MPG^2$]')
		plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
		plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
		plt.legend()
		plt.ylim([0,1])

		plt.show()