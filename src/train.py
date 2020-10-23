
from sys import argv

from common import *
from MinesweeperClass import *

from numpy.random import shuffle

from keras.models import Model, save_model
from keras.layers import Input, Dropout, Activation, Conv2D, Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


def build_2d_model(filters, kernels, blocks):
	"""Build a model to play minesweeper!"""
	
	# Here, we're taking a 3D input in the form of a 2D grid with multiple channels
	# The channels are my way of representing discrete numbers to the model (how many nearby cells are mined?)
	inputs = Input(shape=(GRID_R, GRID_C, CHANNELS))

	# Start by projecting into a different 3D space so we can start using residual connections right away
	x = Conv2D(filters=filters, kernel_size=kernels, padding='same')(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.2)(x)
	
	# Do this over and over... the Add() is our residual connection
	for _ in range(blocks):
		y = Conv2D(filters=filters, kernel_size=kernels, padding='same')(x)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)
		y = Dropout(0.2)(y)

		y = Conv2D(filters=filters, kernel_size=kernels, padding='same')(y)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)
		y = Dropout(0.2)(y)
		x = Add()([x, y])

	# Project into a space that "squeezes" to 2D and apply a sigmoid to map from 0 to 1
	# This 0 to 1 mapping works well with the binary_crossentropy loss function, because...
	# We're predicting which cells are safe and that's more of a multi-out regression than a classification
	outputs = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)

	model = Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model


def compact_frame_to_convolutional_features(string, shape):
	"""Convert a puzzle from its string representation to a numpy array of features"""

	rows, cols = shape

	features = np.zeros((rows, cols, CHANNELS), dtype=np.uint8)
	
	for r in range(rows):
		for c in range(cols):
			if string[r*cols + c].isdigit():
				features[r][c][int(string[r*cols + c])] = 1
	
	return features


def compact_frame_to_convolutional_labels(string, shape):
	"""Convert a puzzle from its string representation to a numpy array of labels"""

	rows, cols = shape

	labels = np.zeros((rows, cols, 1), dtype=np.uint8)
	
	for r in range(rows):
		for c in range(cols):
			if string[r*cols + c] == 's':
				labels[r][c][0] = 1
	
	return labels


def frames_to_dataset(dataset, shape):
	"""Convert puzzle strings into state-action pairs for training a model"""

	features = np.zeros((len(dataset), shape[0], shape[1], CHANNELS), dtype=np.uint8)
	labels = np.zeros((len(dataset), shape[0], shape[1], 1), dtype=np.uint8)

	for d in range(len(dataset)):
		features[d] = compact_frame_to_convolutional_features(dataset[d], shape)
		labels[d] = compact_frame_to_convolutional_labels(dataset[d], shape)

	return features, labels


def train_model_from_file(training_datafile, validation_datafile, model, shape):
	"""Load training and validation data from designated files and train a model to predict safe moves"""

	training_dataset = [x for x in open(training_datafile, 'r').read().split('\n') if len(x) == shape[0]*shape[1]]
	print(len(training_dataset), 'items loaded into training dataset')

	validation_dataset = [x for x in open(validation_datafile, 'r').read().split('\n') if len(x) == shape[0]*shape[1]]
	print(len(validation_dataset), 'items loaded into validation dataset')

	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['accuracy'])

	batch_size = 32
	training_samples = 10000 * batch_size
	validation_samples = training_samples // 10
	history = dict()
	for e in range(1000):

		shuffle(training_dataset)
		shuffle(validation_dataset)

		training_features, training_labels = frames_to_dataset(training_dataset[:training_samples], shape)
		validation_features, validation_labels = frames_to_dataset(validation_dataset[:validation_samples], shape)

		instance = model.fit(
			training_features, training_labels, batch_size=batch_size, epochs=1,
			verbose=1, validation_data=(validation_features, validation_labels))

		for key, value in instance.history.items():
			if key in history:
				history[key] += value
			else:
				history[key] = value

		for key, values in history.items():
			print(key + ' ' + ' '.join([str(x) for x in values]))

		filename = 'minesweeper model '+str(shape[0])+'x'+str(shape[1])+'x'+str(MIN_MINES)+' '+str(e)
		save_model(model, filename, include_optimizer=False, save_format='h5')


if len(argv) == 3:
	train_model_from_file(argv[1], argv[2], build_2d_model(32, (3, 3), 10), (GRID_R, GRID_C))
else:
	print('Usage: python train.py [training dataset path] [validation dataset path]')