import tensorflow as tf
import utils
import numpy as np


class Rnn():
	"""Recurrent neural network"""
	def __init__(self, time_steps = 10):
		self.time_steps = time_steps
		self.model = self.model_definition()


	def model_definition(self):
		return tf.keras.Sequential([
			tf.keras.layers.SimpleRNN(30, input_shape = (self.time_steps, 1), return_sequences = True),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 1000, batch_size = 10):
		optimizer = tf.keras.optimizers.RMSprop(0.001)

		self.model.compile(loss='mse',
				optimizer=optimizer,
				metrics=['mse'])

		x = np.resize(trainset, (batch_size, self.time_steps + 1, 1))
		y = x[:, -1, :]
		x = x[:, :-1, :]
		history = self.model.fit(x, y, epochs=epochs, validation_split = 0.2, verbose=0)

		utils.plot_history(history)


	def project(self, past):
