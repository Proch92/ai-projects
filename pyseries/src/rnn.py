import tensorflow as tf
tf.enable_eager_execution()

import utils
import numpy as np
import os


class Rnn():
	"""Recurrent neural network"""
	def __init__(self, time_steps = 10):
		self.time_steps = time_steps
		self.checkpoint_dir = './training_checkpoints'
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")


	def model_definition(self, batch_size):
		"""[batch_size, time_steps] -> [batch_size, time_steps, 1] ------- '1' is the number of features. we only have one"""
		"""Reshape first argument takes the new shape excluding the batch size -------- (time_steps, 1) ---> [batch_size, time_steps, 1]"""
		return tf.keras.Sequential([
			tf.keras.layers.Reshape((self.time_steps, 1), batch_input_shape = [batch_size, self.time_steps]),
			tf.keras.layers.SimpleRNN(30, return_sequences = True, stateful = True),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 10, batch_size = 10):
		model = self.model_definition(batch_size)
		model.compile(loss='mse',
				optimizer=tf.train.RMSPropOptimizer(0.001),
				metrics=['mse'])

		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix, save_weights_only=True)

		input_sequences = len(trainset) // self.time_steps
		sequences = self.preprocess_data(trainset, batch_size)
		
		history = model.fit(sequences.repeat(), epochs=epochs, steps_per_epoch=(input_sequences // batch_size), verbose=0, callbacks=[checkpoint_callback])

		utils.plot_history(history)


	def evaluate(self, testset):
		model = self.model_definition(1)
		model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
		model.build(tf.TensorShape([1, self.time_steps]))


	def preprocess_data(self, raw_data, batch_size):
		dataset = tf.data.Dataset.from_tensor_slices(raw_data)
		sequences = dataset.batch(self.time_steps+1, drop_remainder=True)
		sequences = sequences.map(lambda sequence: (sequence[:-1], sequence[1:]))
		return sequences.shuffle(10000).batch(batch_size, drop_remainder=True)