import tensorflow as tf
tf.enable_eager_execution()

import datautils
import utils
import numpy as np
import os


class Rnn():
	"""Recurrent neural network"""
	def __init__(self):
		self.checkpoint_dir = './training_checkpoints'
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")


	def model_definition(self, batch_size, time_steps):
		"""[batch_size, time_steps] -> [batch_size, time_steps, 1] ------- '1' is the number of features. we only have one"""
		"""Reshape first argument takes the new shape excluding the batch size -------- (time_steps, 1) ---> [batch_size, time_steps, 1]"""
		return tf.keras.Sequential([
			tf.keras.layers.Reshape((time_steps, 1), batch_input_shape = [batch_size, time_steps]),
			tf.keras.layers.SimpleRNN(30, recurrent_initializer='glorot_uniform', return_sequences = True, stateful = True),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, epochs = 10, batch_size = 10, time_steps = 10):
		model = self.model_definition(batch_size, time_steps)
		model.compile(loss='mse',
				optimizer=tf.train.RMSPropOptimizer(0.001),
				metrics=['mse'])

		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix, save_weights_only=True)

		input_sequences = len(trainset) // time_steps
		sequences = self.preprocess_data(trainset, batch_size, time_steps)
		
		history = model.fit(sequences.repeat(), epochs=epochs, steps_per_epoch=(input_sequences // batch_size), verbose=0, callbacks=[checkpoint_callback])

		utils.plot_history(history)


	def evaluate(self, testset, projection_length=100):
		time_steps = len(testset)

		"""[len(testset)] ---> [1, len(testset)] ---------- adds batch dimension (only 1 batch)"""
		input_tensor = tf.expand_dims(testset, 0)
		input_tensor = tf.to_float(input_tensor)

		model = self.model_definition(1, time_steps)
		model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
		model.build(tf.TensorShape([1, time_steps]))

		results = []
		model.reset_states()

		for i in range(projection_length):
			prediction = model(input_tensor)
			results.append(tf.squeeze(prediction).numpy()) # remove all one-dimensions
			input_tensor = tf.squeeze(prediction, [2]) # remove last dimension (the one created by the Reshape layer)

		return datautils.streamline(results)


	def preprocess_data(self, raw_data, batch_size, time_steps):
		dataset = tf.data.Dataset.from_tensor_slices(raw_data)
		sequences = dataset.batch(time_steps+1, drop_remainder=True)
		sequences = sequences.map(lambda sequence: (sequence[:-1], sequence[1:]))
		return sequences.shuffle(10000).batch(batch_size, drop_remainder=True)