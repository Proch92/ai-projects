import tensorflow as tf
import utils
import datautils


class Fnn():
	"""Simple forward neural network"""
	def __init__(self):


	def model_definition(self, window_size):
		return tf.keras.Sequential([
			tf.keras.layers.Dense(20, activation=tf.nn.tanh, input_shape = [self.window_size]),
			tf.keras.layers.Dense(10, activation=tf.nn.tanh),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 1000, batch_size = 10, window_size = 3):
		model = self.model_definition(window_size)
		self.model.compile(loss='mse',
				optimizer=tf.keras.optimizers.RMSprop(0.001),
				metrics=['mse'])

		dataset = tf.data.Dataset.from_tensor_slices(trainset)
		dataset = dataset.window(window_size)
		dataset = dataset.shuffle(10000)

		history = self.model.fit(dataset.repeat(), epochs=epochs, validation_split = 0.2, verbose=0)

		utils.plot_history(history)
