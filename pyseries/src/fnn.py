import tensorflow as tf
import utils


class Fnn():
	"""Simple forward neural network"""
	def __init__(self, window_size = 3):
		self.window_size = window_size
		self.model = self.model_definition()


	def model_definition(self):
		return tf.keras.Sequential([
			tf.keras.layers.Dense(20, activation=tf.nn.tanh, input_shape = [self.window_size]),
			tf.keras.layers.Dense(10, activation=tf.nn.tanh),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 1000, batch_size = 10):
		optimizer = tf.keras.optimizers.RMSprop(0.001)

		self.model.compile(loss='mse',
				optimizer=optimizer,
				metrics=['mse'])

		x, y = dataset.sliding_window(trainset, self.window_size)
		history = self.model.fit(x, y, epochs=epochs, validation_split = 0.2, verbose=0)

		utils.plot_history(history)
