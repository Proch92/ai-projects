import tensorflow as tf
import utils


class Rnn():
	"""Recurrent neural network"""
	def __init__(self, window_size = 10):
		self.window_size = window_size
		self.model = self.model_definition()


	def model_definition(self):
		return tf.keras.Sequential([
			tf.keras.layers.SimpleRnn(30, activation=tf.nn.tanh, input_shape = [self.window_size]),
			tf.keras.layers.Dense(15, activation=tf.nn.relu),
			tf.keras.layers.Dense(1)
		])


	def train(self, dataset, epochs = 1000, batch_size = 10):
		optimizer = tf.keras.optimizers.RMSprop(0.001)

		self.model.compile(loss='mse',
				optimizer=optimizer,
				metrics=['mae', 'mse'])

		x, y = dataset.sliding_window(self.window_size)
		history = self.model.fit(x, y, epochs=epochs, validation_split = 0.2, verbose=0)

		utils.plot_history(history)
