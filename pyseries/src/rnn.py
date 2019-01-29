import tensorflow as tf
import utils
import numpy as np
import os


class Rnn():
	"""Recurrent neural network"""
	def __init__(self, time_steps = 10, batch_size = 10):
		self.time_steps = time_steps
		self.batch_size = batch_size
		self.model = self.model_definition()


	def model_definition(self):
		return tf.keras.Sequential([
			tf.keras.layers.SimpleRNN(30, batch_input_shape = (self.batch_size, self.time_steps, 1), return_sequences = True, stateful = True),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 10):
		optimizer = tf.keras.optimizers.RMSprop(0.001)

		self.model.compile(loss='mse',
				optimizer=optimizer,
				metrics=['mse'])

		checkpoint_dir = './training_checkpoints'
		checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

		x = trainset[:-1]
		y = trainset[1:]
		dataset = tf.data.Dataset.from_tensor_slices((x, y))
		input_sequences = len(x) // self.time_steps
		dataset = dataset.shuffle(10000).batch(self.batch_size, drop_remainder=True)
		
		history = self.model.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=(input_sequences // self.batch_size), verbose=0, callbacks=[checkpoint_callback])

		utils.plot_history(history)
