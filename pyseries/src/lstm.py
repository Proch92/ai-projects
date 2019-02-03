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


	def model_definition(self, batch_size):
		return tf.keras.Sequential([
			tf.keras.layers.LSTM(
					100,
					return_sequences = True,
					stateful = True,
					batch_input_shape = (batch_size, None, 1)),
			tf.keras.layers.LSTM(
					50,
					return_sequences = True,
					stateful = True),
			tf.keras.layers.Dense(1)
		])


	def train(self, trainset, validationset, epochs, batch_size, time_steps):
		num_sequences = len(trainset) // time_steps
		sequences = self.preprocess_data(trainset, batch_size, time_steps)
		num_val_sequences = len(validationset) // time_steps
		val_sequences = self.preprocess_data(validationset, batch_size, time_steps)

		model = self.model_definition(batch_size)
		model.compile(loss='mse',
				optimizer=tf.train.RMSPropOptimizer(0.001),
				metrics=['mse'])
		model.summary()
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix, save_weights_only=True)
		
		model.reset_states()
		history = model.fit(sequences.repeat(),
				epochs=epochs,
				steps_per_epoch=(num_sequences // batch_size),
				shuffle = False,
				verbose=1,
				callbacks=[checkpoint_callback],
				validation_data=val_sequences.repeat(),
				validation_steps=(num_val_sequences // batch_size))

		utils.plot_history(history)


	def evaluate(self, testset, projection_length):
		input_tensor = tf.expand_dims(testset, 1) # adds features dimension
		input_tensor = tf.expand_dims(input_tensor, 0) # adds batch dimension
		input_tensor = tf.to_float(input_tensor)

		model = self.model_definition(1)
		model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
		model.build(tf.TensorShape([1, None]))
		model.summary()

		projection = []
		model.reset_states()

		guided = model(input_tensor)
		input_tensor = guided[:,-1:,:]
		guided = tf.squeeze(guided).numpy()

		for i in range(projection_length):
			output = model(input_tensor)
			prediction = output[:,-1:,:]
			projection.append(tf.squeeze(prediction).numpy())
			input_tensor = prediction

		return (guided, projection)


	def preprocess_data(self, raw_data, batch_size, time_steps):
		reshaped = tf.expand_dims(raw_data, 1)
		dataset = tf.data.Dataset.from_tensor_slices(reshaped)
		sequences = dataset.batch(time_steps+1, drop_remainder=True)
		sequences = sequences.map(lambda sequence: (sequence[:-1], sequence[1:]))
		return sequences.batch(batch_size, drop_remainder=True)