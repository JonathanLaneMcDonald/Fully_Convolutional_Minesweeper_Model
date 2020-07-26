
import os
import time
import numpy as np
from math import log

from copy import copy
from common import *
from MinesweeperClass import *

from numpy.random import random as npr
from numpy.random import shuffle

from keras.models import Model, load_model
from keras.layers import Input, Dropout, Activation, Conv2D, Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def build_2d_model(filters, kernels, layers):
	
	input = Input(shape=(GRID_R,GRID_C,CHANNELS))

	x = Conv2D(filters=filters, kernel_size=kernels, padding='same')(input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.2)(x)
	
	for _ in range(layers-1):
		y = Conv2D(filters=filters, kernel_size=kernels, padding='same')(x)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)
		y = Dropout(0.2)(y)
		x = Add()([x,y])

	output = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')(x)

	model = Model(inputs=input, outputs=output)
	model.summary()
	return model

def compact_frame_to_convolutional_features(string, shape):
	rows, cols = shape

	features = np.zeros((rows, cols, CHANNELS), dtype=np.uint8)
	
	for r in range(rows):
		for c in range(cols):
			if string[r*cols + c].isdigit():
				features[r][c][int(string[r*cols + c])] = 1
	
	return features

def compact_frame_to_convolutional_labels(string, shape):
	rows, cols = shape

	labels = np.zeros((rows, cols, 1), dtype=np.uint8)
	
	for r in range(rows):
		for c in range(cols):
			if string[r*cols + c] == 's':
				labels[r][c][0] = 1
	
	return labels

def frames_to_dataset(dataset, shape):
	features = np.zeros((len(dataset), shape[0], shape[1], CHANNELS), dtype=np.uint8)
	labels = np.zeros((len(dataset), shape[0], shape[1], 1), dtype=np.uint8)

	for d in range(len(dataset)):
		features[d] = compact_frame_to_convolutional_features(dataset[d], shape)
		labels[d] = compact_frame_to_convolutional_labels(dataset[d], shape)

	return features, labels

def train_model_from_file(training_datafile, validation_datafile, model, shape):
	training_dataset = [x for x in open(training_datafile,'r').read().split('\n') if len(x) == shape[0]*shape[1]]
	print (len(training_dataset),'items loaded into training dataset')

	validation_dataset = [x for x in open(validation_datafile,'r').read().split('\n') if len(x) == shape[0]*shape[1]]
	print (len(validation_dataset),'items loaded into validation dataset')

	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002), metrics=['accuracy'])

	batch_size = 32
	training_samples = 10000 * batch_size
	validation_samples = training_samples // 10
	history = dict()
	for e in range(1000):

		shuffle(training_dataset)
		shuffle(validation_dataset)

		training_features, training_labels = frames_to_dataset(training_dataset[:training_samples], shape)
		validation_features, validation_labels = frames_to_dataset(validation_dataset[:validation_samples], shape)

		instance = model.fit(training_features, training_labels, batch_size=batch_size, epochs=1, verbose=1, validation_data=(validation_features, validation_labels))

		for key, value in instance.history.items():
			if key in history:
				history[key] += value
			else:
				history[key] = value

		for key, values in history.items():
			print (key + ' ' + ' '.join([str(x) for x in values]))

		model.save('debug model '+str(shape[0])+'x'+str(shape[1])+'x'+str(MIN_MINES)+' '+str(e))

train_model_from_file('training', 'validation', build_2d_model(32, (3,3), 20), (GRID_R, GRID_C))
