
import os
import time
import numpy as np
from math import log

from copy import copy
from MinesweeperClass import *

from numpy.random import random as npr
from numpy.random import shuffle

if __name__ == '__main__':
	from keras.models import Sequential, Model, load_model
	from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
	from keras.layers import Conv2D, Conv3D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
	from keras.optimizers import Adam, Adagrad, Adadelta
	from keras.utils import to_categorical
	from keras.layers.merge import add, Add
	from keras.layers.normalization import BatchNormalization

# **********************************************************************
# *****************************evaluation*******************************
# **********************************************************************

def build_difficulty_stats(R, C, G):
	for m in range(MIN_MINES,MAX_MINES+1):
		wins = 0
		losses = 0
		for i in range(G):
			game = Minesweeper( R, C, m )

			while game.get_game_status() == game.INPROGRESS:
				unvisited_cells = game.get_moves()
				selection = int(npr()*len(unvisited_cells))
				
				r = unvisited_cells[selection][0]
				c = unvisited_cells[selection][1]

				game.visit_cell((r,c))

			if game.get_game_status() == game.VICTORY:
				wins += 1
			elif game.get_game_status() == game.DEFEAT:
				losses += 1
			
		print (m,'mines',wins,'wins',losses,'losses')


# **********************************************************************
# *************************constants************************************
# **********************************************************************

GENERATE_DATASET = 0
BUILD = 1
EXPLORE = 2
PLAY = 3
EVALUATE = 4
RL = 5
TRAIN_FROM_FILE = 6

# 9 channels for the proximity map, 1 for the visibility map
CHANNELS = 10

"""
Board Sizes:
	beginner 		8x8 with 10 mines		15.625% mines
	intermediate	16x16 with 40 mines		15.625% mines
	expert			16x30 with 99 mines		~20% mines
	
	easy			16x32 with 64 mines		12.50% mines
	medium			16x32 with 96 mines		18.75% mines
	hard			16x32 with 128 mines	25.00% mines
"""

GRID_R = 16
GRID_C = 30
GRID_CELLS = GRID_R * GRID_C

MIN_MINES = 99
MAX_MINES = MIN_MINES

#selection = BUILD
#selection = EXPLORE
#selection = PLAY
#selection = EVALUATE
#selection = RL
selection = TRAIN_FROM_FILE
training_file = 'training'
validation_file = 'validation'













# **********************************************************************
# **********************neural network utils****************************
# **********************************************************************

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

def board_to_features(game):
	rows = game.rows
	cols = game.cols
	channels = CHANNELS

	new_array = np.zeros((rows, cols, channels), dtype=np.uint8)

	proximal = game.get_proximity_field()
	visible = game.get_visible_field()

	for r in range(game.rows):
		for c in range(game.cols):
			if visible[r][c]:
				new_array[r][c][proximal[r][c]] = 1

	return new_array

def prepare_training_labels(model, training_features, unfinished_labels):
	# first, let's see what the network currently thinks about this board position
	training_labels = model.predict(training_features, verbose=1)

	# now let's give it a nudge by encouraging or discouraging this particular behavior
	for i in range(len(unfinished_labels)):
		index = abs(unfinished_labels[i])-1
		if unfinished_labels[i] < 0:
			training_labels[i][index] = -1
		else:
			training_labels[i][index] = 1

	return training_labels

def generate_heat_map(game, model):
	input_features = np.array(board_to_features(game), dtype=np.intc).reshape(1, GRID_R, GRID_C, CHANNELS)
	prediction = model.predict(input_features).reshape(GRID_CELLS)

	flag_field = game.get_flagged_field()

	heat_map = []
	for i in range(len(prediction)):
		r = i//GRID_C
		c = i%GRID_C
		if flag_field[r][c]:
			# FIXME: don't change this value here.  check for flags somewhere else because this compromises the goal of the function
			heat_map.append((float(-1),(r,c)))
		else:
			heat_map.append((float(prediction[i]),(r,c)))

	return heat_map

def select_top_moves(game, model, moves_requested):
	heat_map = generate_heat_map(game, model)
	heat_map.sort()
	heat_map.reverse()
	
	# just start by picking the top preferences
	top_moves = []
	possible_moves = game.get_visible_field()
	for i in heat_map:
		r = i[1][0]
		c = i[1][1]
		if possible_moves[r][c] == 0:
			top_moves.append((r,c))
			if len(top_moves) == moves_requested:
				return top_moves
	return top_moves

# **********************************************************************
# ***********************simple logic functions*************************
# **********************************************************************

def evaluate(target_games, model, mines=MIN_MINES, nn_predicts_opening_move=False):
	deaths = 0
	moves_played = 0
	games_played = 0
	games_won = 0
	global_likelihood = 0
	while games_played < target_games:
		game = Minesweeper( GRID_R, GRID_C, mines )
		games_played += 1

		likelihood = 1.0
		to_smitherines = False
		while game.get_game_status() == game.INPROGRESS:
			possible_moves = game.get_moves()
			safe_moves = [x for x in possible_moves if not game.this_cell_is_mined(x)]

			if len(safe_moves):
				if game.first_move_has_been_made() or nn_predicts_opening_move:
					moves_played += 1
					selected_move = select_top_moves(game, model, 1)[0]
					if game.this_cell_is_mined(selected_move):
						game.forfeit_game()
						to_smitherines = True
						deaths += 1
					else:
						likelihood *= len(safe_moves) / len(possible_moves)
					game.visit_cell(selected_move)
				else:
					game.visit_cell(safe_moves[int(npr()*len(safe_moves))])
			else:
				game.forfeit_game()# because we're surrounded by mines ;(

		global_likelihood += log(likelihood)/log(10)
		if not to_smitherines:
			games_won += 1

		if games_played % 1 == 0:
			print (games_played, float(deaths)/moves_played, float(games_won)/games_played, float(moves_played)/games_played, float(global_likelihood)/games_played)

	return float(deaths)/moves_played, float(games_won)/games_played, float(moves_played)/games_played, float(global_likelihood)/games_played

def compact_frames_to_features(dataset, shape):
	features = np.zeros((len(dataset), shape[0], shape[1], CHANNELS), dtype=np.uint8)

	for d in range(len(dataset)):
		features[d] = compact_frame_to_convolutional_features(dataset[d], shape)

	return features

def compact_frames_to_labels(dataset, shape):
	labels = np.zeros((len(dataset), shape[0], shape[1], 1), dtype=np.uint8)

	for d in range(len(dataset)):
		labels[d] = compact_frame_to_convolutional_labels(dataset[d], shape)

	return labels

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
		training_features = compact_frames_to_features(training_dataset[:training_samples], shape)
		training_labels = compact_frames_to_labels(training_dataset[:training_samples], shape)

		shuffle(validation_dataset)
		validation_features = compact_frames_to_features(validation_dataset[:validation_samples], shape)
		validation_labels = compact_frames_to_labels(validation_dataset[:validation_samples], shape)

		instance = model.fit(training_features, training_labels, batch_size=batch_size, epochs=1, verbose=1, validation_data=(validation_features, validation_labels))

		for key, value in instance.history.items():
			if key in history:
				history[key] += value
			else:
				history[key] = value

		for key, values in history.items():
			print (key + ' ' + ' '.join([str(x) for x in values]))

		model.save('debug model '+str(shape[0])+'x'+str(shape[1])+'x'+str(MIN_MINES)+' '+str(e))

if __name__ == '__main__':
	if selection == TRAIN_FROM_FILE:
		train_model_from_file(training_file, validation_file, build_2d_model(32, (3,3), 10), (GRID_R, GRID_C))
	elif selection == BUILD:
		build_difficulty_stats( GRID_R, GRID_C, 10000 )
	elif selection == EVALUATE:
		evaluate(1000, load_model('debug model 16x30x99 3 - 3d model'))
