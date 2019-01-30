import tensorflow as tf
tf.enable_eager_execution()

import datautils
import utils
import numpy as np
import os


class LSTM():
	"""Recurrent neural network"""
	def __init__(self):
		self.checkpoint_dir = './training_checkpoints'
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")


	def model_definition(self, batch_size, time_steps):
		return tf.keras.Sequential([
			tf.keras.layers.LSTM(
					30,
					recurrent_initializer='glorot_uniform',
					return_sequences = True,
					stateful = True,
					batch_input_shape = (batch_size, time_steps, 1)),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 10, batch_size = 10, time_steps = 10):
		input_sequences = len(trainset) // time_steps
		sequences = self.preprocess_data(trainset, batch_size, time_steps)

		model = self.model_definition(batch_size, time_steps)
		model.compile(loss='mse',
				optimizer=tf.train.RMSPropOptimizer(0.001),
				metrics=['mse'])
		model.summary()
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix, save_weights_only=True)
		
		history = model.fit(sequences.repeat(), epochs=epochs, steps_per_epoch=(input_sequences // batch_size), verbose=0, callbacks=[checkpoint_callback])

		utils.plot_history(history)


	def evaluate(self, testset, projection_length=100):
		time_steps = len(testset)

		input_tensor = tf.expand_dims(testset, 1) # adds fetures dimension
		input_tensor = tf.expand_dims(input_tensor, 0) # adds batch dimension
		input_tensor = tf.to_float(input_tensor)

		model = self.model_definition(1, time_steps)
		model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
		model.build(tf.TensorShape([1, time_steps]))

		results = []
		model.reset_states()

		for i in range(projection_length):
			prediction = model(input_tensor)
			results.append(tf.squeeze(prediction).numpy()) # remove all one-dimensions
			input_tensor = prediction

		return datautils.streamline(results)


	def preprocess_data(self, raw_data, batch_size, time_steps):
		reshaped = tf.expand_dims(raw_data, 1)
		dataset = tf.data.Dataset.from_tensor_slices(reshaped)
		sequences = dataset.batch(time_steps+1, drop_remainder=True)
		sequences = sequences.map(lambda sequence: (sequence[:-1], sequence[1:]))
		return sequences.shuffle(10000).batch(batch_size, drop_remainder=True)