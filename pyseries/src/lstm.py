import tensorflow as tf
tf.enable_eager_execution()

import datautils
import utils
import numpy as np
import os


class LSTM():
	"""Recurrent neural network"""
	def __init__(self):
		pass


	def model_definition(self, batch_size):
		return tf.keras.Sequential([
			tf.keras.layers.LSTM(
					120,
					#40,
					dropout=0.2,
					recurrent_dropout=0.2,
					return_sequences = True,
					stateful = True,
					batch_input_shape = (batch_size, None, 1)),
			tf.keras.layers.LSTM(
					60,
					#20,
					dropout=0.2,
					recurrent_dropout=0.2,
					return_sequences = True,
					stateful = True),
			tf.keras.layers.Dense(1)
		])


	def train(self, modelname, trainset, epochs, batch_size, time_steps):
		(x, y) = self.prepare_data(trainset, batch_size, time_steps)

		model = self.model_definition(batch_size)
		model.compile(loss='mse',
				optimizer=tf.train.RMSPropOptimizer(0.001),
				metrics=['mse'])
		model.summary()
		
		model.reset_states()
		history = model.fit(
				x, 
				y,
				epochs=epochs,
				batch_size=batch_size,
				shuffle = False,
				verbose=1)

		model.save(os.path.join('models', modelname + '.h5'))

		return history


	def evaluate(self, modelname, testset, projection_length):
		input_tensor = tf.expand_dims(testset, 1) # adds features dimension
		input_tensor = tf.expand_dims(input_tensor, 0) # adds batch dimension
		input_tensor = tf.to_float(input_tensor)

		model = self.model_definition(1) # batch size = 1
		model.load_weights(os.path.join('models', modelname + '.h5'))
		model.build(tf.TensorShape([1, None]))

		projection = []
		model.reset_states()

		# forward passing in the entire test test to build a good memory state
		guided = model(input_tensor)
		input_tensor = guided[:,-1:,:]
		projection.append(tf.squeeze(input_tensor).numpy())

		for i in range(projection_length - 1):
			output = model(input_tensor)
			prediction = output[:,-1:,:]
			projection.append(tf.squeeze(prediction).numpy())
			input_tensor = prediction

		return projection


	#####################################################
	# 	(datalen)										#
	# duplicate sequences batch_size times				#
	# 	(batch_size, datalen)							#
	# random rolls										#
	# discard rolled points								#
	# 	(batch_size, time_steps * num_cuts + 1)			#
	# expand dimension for LSTM							#
	#	(batch_size, time_steps * num_cuts + 1, 1)		#
	# decouple x and y									#
	#	(batch_size, times_steps * num_cuts, 1)			#
	# cut sequences into batches of time_steps length	#
	#	(num_cuts * batch_size, times_steps, 1)			#
	#####################################################

	def prepare_data(self, data, batch_size, time_steps):
		print("datalen: {}, batch_size: {}, time_steps: {}".format(len(data), batch_size, time_steps))
		
		# data augmentation. duplicating the sequence and rolling it to change the starting point. then batch-train all the sequences
		sequences = [data for _ in range(batch_size)]
		shifts = np.random.randint(time_steps, size=batch_size)
		sequences = list(map(lambda batch, shift: np.roll(batch, -shift), sequences, shifts))
		
		# discard points that rolled over the end
		num_cuts = (len(data) // time_steps) - 1
		sequences = list(map(lambda batch: batch[:(num_cuts*time_steps)+1], sequences))
		
		# expanding dimension for lstm input space
		sequences = np.expand_dims(sequences, axis=2)

		# decoupling x and y
		(x, y) = (sequences[:, :-1, :], sequences[:, 1:, :])

		# cutting sequences of length time_steps. each cut block will train the model in an epoch
		mapx = map(lambda i: x[:, time_steps*i:time_steps*(i+1), :], range(num_cuts))
		mapy = map(lambda i: y[:, time_steps*i:time_steps*(i+1), :], range(num_cuts))
		x = np.concatenate(list(mapx), axis=0)
		y = np.concatenate(list(mapy), axis=0)

		return (x, y)


	def save(self, filename):
		tf.keras.models.save_model(
			model,
			filepath,
			overwrite=True
		)
