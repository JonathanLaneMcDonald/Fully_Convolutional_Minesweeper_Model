
import os
import time
import numpy as np
from math import log

from copy import copy
from MinesweeperClass import *

from numpy.random import random as npr
from numpy.random import shuffle

if __name__ == '__main__':
	from keras.models import Sequential, model_from_json, Model
	from keras.layers import Dense, Dropout, Activation, Flatten, Input
	from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
	from keras.optimizers import Adam, Adagrad, Adadelta
	from keras.utils import to_categorical
	from keras.layers.merge import add
	from keras.layers.normalization import BatchNormalization

# **********************************************************************
# *********************general utils************************************
# **********************************************************************

from keras.models import model_from_json

def load_model(fname):
	if fname == None:
		return None

	model = model_from_json(open(fname,'r').read())
	model.load_weights(fname+' weights')
	return model

def save_model(filename_base, model):
	open(filename_base,'w').write(model.to_json())
	model.save_weights(filename_base + ' weights')

def randomize(array):
	for i in range(len(array)):
		j = int(npr()*len(array))
		while j == i:
			j = int(npr()*len(array))
		
		temp = array[i]
		array[i] = array[j]
		array[j] = temp
	return array
		
def these_fields_are_the_same(field_a, field_b, shape):
	the_same = True
	rows = shape[0]
	cols = shape[1]
	for r in range(rows):
		for c in range(cols):
			if field_a[r][c] != field_b[r][c]:
				the_same = False
	return the_same

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

GRID_R = 32
GRID_C = 32
GRID_CELLS = GRID_R * GRID_C

MIN_MINES = 128
MAX_MINES = MIN_MINES

selection = GENERATE_DATASET
#selection = BUILD
#selection = EXPLORE
selection = PLAY
#selection = EVALUATE
#selection = RL
#selection = TRAIN_FROM_FILE
training_file = str(GRID_R)+'x'+str(GRID_C)+'x'+str(MIN_MINES)
trained_model = 'debug model (16, 30) 11'

BASIC_SOLVER = False
#BASIC_SOLVER = True

TRAINING_QUOTA = 250000
BATCH = 128













# **********************************************************************
# **********************neural network utils****************************
# **********************************************************************

def build_resnet(filters, kernels, blocks):
	
	input 	= Input(shape=(GRID_R,GRID_C,CHANNELS))

	conv 	= Conv2D(filters=filters, kernel_size=kernels, strides=(1,1), padding="same")(input)
	norm	= BatchNormalization(axis=3)(conv)
	bk_input= Activation("relu")(norm)
	
	for block in range(blocks):
		conv	= Conv2D(filters=filters, kernel_size=kernels, strides=(1,1), padding="same")(bk_input)
		norm	= BatchNormalization(axis=3)(conv)
		relu	= Activation("relu")(norm)

		conv	= Conv2D(filters=filters, kernel_size=kernels, strides=(1,1), padding="same")(relu)
		norm	= BatchNormalization(axis=3)(conv)
		skip	= add([norm, bk_input])
		bk_input= Activation("relu")(skip)

	conv	= Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), padding="same")(bk_input)
	norm	= BatchNormalization(axis=3)(conv)
	relu	= Activation("relu")(norm)

	output	= Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding="same", activation="sigmoid")(relu)

	model = Model(inputs=input, outputs=output)
	model.summary()
	return model

def compact_features_to_features_and_labels(string, shape):
	rows, cols = shape

	features = np.zeros((rows, cols, CHANNELS), dtype=np.uint8)
	labels = np.zeros((rows, cols, 1), dtype=np.uint8)
	
	for r in range(rows):
		for c in range(cols):
			if string[r*cols + c].isdigit():
				features[r][c][int(string[r*cols + c])] = 1
			if string[r*cols + c] == 's':
				labels[r][c][0] = 1
	
	return features, labels

def compact_frame_to_linear_features(string):

	features = np.zeros(len(string) * CHANNELS, dtype=np.uint8)
	
	for p in range(len(string)):
		if string[p].isdigit():
			features[p*CHANNELS + int(string[p])] = 1

	return features

def compact_frame_to_linear_labels(string):

	labels = np.zeros(len(string) * CHANNELS, dtype=np.uint8)
	
	for p in range(len(string)):
		if string[p] == 's':
			labels[p] = 1

	return labels

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

def plant_island_flags(game):
	shape = game.get_field_shape()
	rows = shape[0]
	cols = shape[1]

	vis_map = game.get_visible_field()
	pro_map = game.get_proximity_field()
	
	flags = []
	for R in range(rows):
		for C in range(cols):
			if pro_map[R][C] and vis_map[R][C]:
				neighbors = []
				for r in range(R-1, R+2):
					for c in range(C-1, C+2):
						if game.point_is_on_board(r,c):
							if vis_map[r][c] == 0:
								neighbors.append((r,c))
				if len(neighbors) == pro_map[R][C]:
					for cell in neighbors:
						game.place_flag(cell)
	return game

def visit_reasonable_cells(game):
	shape = game.get_field_shape()
	rows = shape[0]
	cols = shape[1]

	game = plant_island_flags(game)

	vis_map = game.get_visible_field()
	pro_map = game.get_proximity_field()
	flg_map = game.get_flagged_field()
	
	for R in range(rows):
		for C in range(cols):
			if pro_map[R][C] and vis_map[R][C]:
				flagged_neighbors = []
				for r in range(R-1, R+2):
					for c in range(C-1, C+2):
						if game.point_is_on_board(r,c):
							if flg_map[r][c]:
								flagged_neighbors.append((r,c))
				if len(flagged_neighbors) == pro_map[R][C]:
					for r in range(R-1, R+2):
						for c in range(C-1, C+2):
							if game.point_is_on_board(r,c):
								if flg_map[r][c] == 0 and vis_map[r][c] == 0:
									game.visit_cell((r,c))
	return game

def make_logical_inferences(game):
	old_visi_field = game.get_visible_field()
	old_flag_field = game.get_flagged_field()
	game = visit_reasonable_cells(game)
	game = visit_reasonable_cells(game)
	new_visi_field = game.get_visible_field()
	new_flag_field = game.get_flagged_field()
	while not these_fields_are_the_same(old_visi_field, new_visi_field, game.get_field_shape()) and not these_fields_are_the_same(old_flag_field, new_flag_field, game.get_field_shape()):
		old_visi_field = game.get_visible_field()
		old_flag_field = game.get_flagged_field()
		game = visit_reasonable_cells(game)
		game = visit_reasonable_cells(game)
		new_visi_field = game.get_visible_field()
		new_flag_field = game.get_flagged_field()
	return game

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

		if games_played % 10 == 0:
			print (games_played, float(deaths)/moves_played, float(games_won)/games_played, float(moves_played)/games_played, float(global_likelihood)/games_played)

	return float(deaths)/moves_played, float(games_won)/games_played, float(moves_played)/games_played, float(global_likelihood)/games_played

def exploratory_model_training(model):
	pass

def rl_training(model):
	pass

def compact_frames_to_features(dataset, shape, is_convolutional=True):
	features = None

	if is_convolutional:
		features = np.zeros((len(dataset), shape[0], shape[1], CHANNELS), dtype=np.uint8)
	else:
		features = np.zeros((len(dataset), shape[0] * shape[1] * CHANNELS), dtype=np.uint8)

	for d in range(len(dataset)):
		if is_convolutional:
			features[d] = compact_frame_to_convolutional_features(dataset[d], shape)
		else:
			features[d] = compact_frame_to_linear_features(dataset[d])

	return features

def compact_frames_to_labels(dataset, shape, is_convolutional=True):
	labels = None

	if is_convolutional:
		labels = np.zeros((len(dataset), shape[0], shape[1], 1), dtype=np.uint8)
	else:
		labels = np.zeros((len(dataset), shape[0] * shape[1]), dtype=np.uint8)

	for d in range(len(dataset)):
		if is_convolutional:
			labels[d] = compact_frame_to_convolutional_labels(dataset[d], shape)
		else:
			labels[d] = compact_frame_to_linear_labels(dataset[d])

	return labels

def train_model_from_file(filename, model, shape, convolutional_features=True, convolutional_labels=True):
	dataset = [x for x in open(filename,'r').read().split('\n') if len(x) == shape[0]*shape[1]]
	print (len(dataset),'items loaded into dataset')

	lr = 0.002
	samples = 100000
	for e in range(1,1000):

		model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr/e), metrics=['accuracy'])

		shuffle(dataset)
		training_features = compact_frames_to_features(dataset[:samples], shape, convolutional_features)
		training_labels = compact_frames_to_labels(dataset[:samples], shape, convolutional_labels)

		model.fit(training_features, training_labels, epochs=1, verbose=1, validation_split=0.1)

		save_model('debug model '+str(shape[0])+'x'+str(shape[1])+'x'+str(MIN_MINES)+' '+str(e),model)

from tkinter import *

SQ = 24

h='0123456789abcdef'
def itoh(n):
	return h[(n>>20)&0xf]+h[(n>>16)&0xf]+h[(n>>12)&0xf]+h[(n>>8)&0xf]+h[(n>>4)&0xf]+h[(n>>0)&0xf]

class display( Frame ):

	def draw(self):
		self.canvas.delete('all')
		
		prediction = None
		if len(self.model):
			if self.model_selection < len(self.model):
				prediction = generate_heat_map(self.game, self.model[self.model_selection][0])

		mine_field = self.game.get_mine_field()
		flag_field = self.game.get_flagged_field()
		prox_field = self.game.get_proximity_field()
		visi_field = self.game.get_visible_field()
		bord_field = self.game.get_border_field()

		(game_rows, game_cols) = self.game.get_field_shape()
		for r in range(game_rows):
			for c in range(game_cols):
				
				font = ('TakaoMincho', 16)

				fill = 'white'
				outline = 'black'
				if flag_field[r][c]:
					self.canvas.create_rectangle(1+c*SQ,1+r*SQ,1+(c+1)*SQ,1+(r+1)*SQ,fill='yellow',outline=outline)
					fill = 'yellow'
				elif visi_field[r][c]:
					if mine_field[r][c]:
						self.canvas.create_rectangle(1+c*SQ,1+r*SQ,1+(c+1)*SQ,1+(r+1)*SQ,fill='red',outline=outline)
					else:
						self.canvas.create_rectangle(1+c*SQ,1+r*SQ,1+(c+1)*SQ,1+(r+1)*SQ,fill='white',outline=outline)
						if prox_field[r][c]:
							self.canvas.create_text(c*SQ+SQ/2,r*SQ+SQ/2,font=font,text=str(prox_field[r][c]))
				else:
					self.canvas.create_rectangle(1+c*SQ,1+r*SQ,1+(c+1)*SQ,1+(r+1)*SQ,fill='grey',outline=outline)
					fill = 'grey'

				# these cells are 'border cells'
				if bord_field[r][c]:
					self.canvas.create_rectangle(2+c*SQ,2+r*SQ,(c+1)*SQ,(r+1)*SQ,fill=fill,outline='yellow')

				# this is where the 'cursor' is
				if r == self.row and c == self.col:
					self.canvas.create_rectangle(2+c*SQ,2+r*SQ,(c+1)*SQ,(r+1)*SQ,fill=fill,outline='red')
					if visi_field[r][c]:
						if mine_field[r][c]:
							self.canvas.create_rectangle(1+c*SQ,1+r*SQ,1+(c+1)*SQ,1+(r+1)*SQ,fill='red',outline=outline)
						elif prox_field[r][c]:
							self.canvas.create_text(c*SQ+SQ/2,r*SQ+SQ/2,font=font,text=str(prox_field[r][c]))

		if prediction:
			# find the minimum and maximum extents of the prediction values - this way, we can assign percentage values to cells
			sum_over_interesting = []
			for p in prediction:
				r = p[1][0]
				c = p[1][1]
				
				if visi_field[r][c] == 0 and flag_field[r][c] == 0:
					sum_over_interesting.append(p[0])

			if len(sum_over_interesting):
				#print sum_over_interesting
				coldest = min(sum_over_interesting)
				warmest = max(sum_over_interesting)
				t_range = warmest - coldest
				average = float(t_range)/2

				visualize = []
				if t_range:
					for p in prediction:
						r = p[1][0]
						c = p[1][1]

						value = 0
						if coldest <= p[0] <= warmest:
							value = float(p[0] - coldest) / t_range

						visualize.append((value,(r,c)))

				for v in visualize:
					value = 2*v[0] - 1

					red = 255
					green = 255
					blue = 255

					if value < 0:
						red -= int(128*-value)
						green -= int(128*-value)
					else:
						blue -= int(128*value)
						green -= int(128*value)
					
					if red < 0:		red = 0
					if green < 0:	green = 0
					if blue < 0:	blue = 0	

					color = (red<<16) + (green<<8) + blue

					r = v[1][0]
					c = v[1][1]

					if visi_field[r][c]:
						self.canvas.create_oval(3+c*SQ,3+r*SQ,c*SQ+SQ/3,r*SQ+SQ/3,fill='white')
					else:
						self.canvas.create_oval(3+c*SQ,3+r*SQ,c*SQ+SQ/3,r*SQ+SQ/3,fill='#'+itoh(color))

						#if self.game.get_game_status() == self.game.INPROGRESS:
						#	write_value = abs(value)

						#	self.canvas.create_text(43+c*SQ,13+r*SQ,text=str(write_value)[:5])

					if r == self.row and c == self.col:
						print ('Model Predicts:',value)

		if self.showing_mines:
			for r in range(game_rows):
				for c in range(game_cols):
					if mine_field[r][c]:
						self.canvas.create_oval(c*SQ+SQ/4,r*SQ+SQ/4,c*SQ+3*SQ/4,r*SQ+3*SQ/4,fill='red')

		self.canvas.update_idletasks()
		#time.sleep(0.2)

	def load_models(self):
		files = os.listdir('./')
		files.sort()
		
		model_pairs = []
		for i in files:
			if i.find('model') != -1:
				if os.path.exists(i+' weights'):
					model_pairs.append((i,i+' weights'))
		model_pairs.sort()
		
		self.model = []
		self.model_selection = 0

		for i in model_pairs:
			new_model = model_from_json(open(i[0],'r').read())
			new_model.load_weights(i[1])
			self.model.append((new_model,i[0]))
			print ('Model loaded',i[0],i[1])
		print (len(self.model),'models loaded')

		self.model_selection = len(self.model) - 1

	def nnsolver(self):
		smitherines = False
		while not smitherines:
			while self.game.get_game_status() == self.game.INPROGRESS:
				safe_moves = [x for x in self.game.get_moves() if not self.game.this_cell_is_mined(x)]

				if len(safe_moves):
					if self.game.first_move_has_been_made():
						selected_move = select_top_moves(self.game, self.model[self.model_selection][0], 1)[0]
						if self.game.this_cell_is_mined(selected_move):
							smitherines = True
							self.game.forfeit_game()
						else:
							self.game.place_flag(selected_move)
							self.draw()
							self.game.remove_flag(selected_move)

						self.game.visit_cell(selected_move)
					else:
						self.game.visit_cell(safe_moves[int(npr()*len(safe_moves))])
				else:
					self.game.forfeit_game()# because we're surrounded by mines ;(

				self.draw()
			smitherines = True
			if not smitherines:
				self.game = Minesweeper(GRID_R, GRID_C, MIN_MINES)			

	def keyboard(self, event):
		if event.keysym == 'Escape':		self.quit()

		if event.keysym == 'Up':			self.row -= 1
		if event.keysym == 'Down':			self.row += 1
		if event.keysym == 'Left':			self.col -= 1
		if event.keysym == 'Right':			self.col += 1

		if self.row < 0:					self.row = 0
		if self.row >= self.rows:			self.row = self.rows-1
		if self.col < 0:					self.col = 0
		if self.col >= self.cols:			self.col = self.cols-1

		if event.char == 'e':				self.mines += 1
		if event.char == 'q':				self.mines -= 1
		if event.char == 'e' or event.char == 'q':	print ('Mines:',self.mines)
		
		if event.char == 'd':				self.cols += 1
		if event.char == 'a':				self.cols -= 1
		if self.cols < 1:					self.cols = 1
		if event.char == 'd' or event.char == 'a':	print ('Columns adjusted to:',self.cols)
		
		if event.char == 'w':				self.rows += 1
		if event.char == 's':				self.rows -= 1
		if self.rows < 1:					self.rows = 1
		if event.char == 'w' or event.char == 's':	print ('Rows adjusted to:',self.rows)

		if event.char == '-':							self.model_selection -= 1
		if event.char == '+':							self.model_selection += 1
		if self.model_selection < 0:					self.model_selection = 0
		if self.model_selection >= len(self.model):		self.model_selection = len(self.model)-1
		if event.char == '-' or event.char == '+':	print ('Model Selection:',self.model[self.model_selection][1])

		if event.keysym == 'g':
			# if you have models loaded, you'll break everything if you don't reset the board size
			if len(self.model):
				self.mines = MIN_MINES
				self.rows = GRID_R
				self.cols = GRID_C

			self.game = Minesweeper(self.rows, self.cols, self.mines)

		if event.keysym == 'l':
			self.load_models()

		if event.keysym == 'L':
			self.load_ensemble()

		if event.keysym == 'E':
			self.test_ensemble()

		if event.keysym == 'm':
			self.showing_mines ^= 1

		message = ''
		if self.game: # don't do game actions when there's no game or you'll break things
			if event.keysym == 's':
				self.nnsolver()

			if event.keysym == 'r':
				self.game.reset()

			if event.keysym == 'h':
				self.game = visit_reasonable_cells(self.game)

			if event.keysym == 'H':
				self.game = make_logical_inferences(self.game)

			if event.keysym == 'space':
				if self.game.is_visible((self.row, self.col)):
					self.game.set_invisible((self.row, self.col))
				else:
					message = 'game.visit_cell('+str(self.row)+','+str(self.col)+')'
					self.game.visit_cell((self.row, self.col))
					#self.game = make_logical_inferences(self.game)

			if event.keysym == 'f':
				if self.game.get_flagged_field()[self.row][self.col]:
					message = 'game.remove_flag('+str(self.row)+','+str(self.col)+')'
					self.game.remove_flag((self.row, self.col))
				else:
					message = 'game.place_flag('+str(self.row)+','+str(self.col)+')'
					self.game.place_flag((self.row, self.col))
			
			if event.keysym == 'b':
				if self.game.get_mine_field()[self.row][self.col]:
					message = 'game.remove_mine('+str(self.row)+','+str(self.col)+')'
					self.game.remove_mine((self.row, self.col))
				else:
					message = 'game.place_mine('+str(self.row)+','+str(self.col)+')'
					self.game.place_mine((self.row, self.col))
			
			if self.game.game_status == self.game.VICTORY:
				for i in range(10):
					print ('yay, victory!')
			elif self.game.game_status == self.game.DEFEAT:
				for i in range(10):
					print ('boo, you lose')

		print (event.keysym, event.char, message)

		self.draw()
	
	def invent_canvas(self):
		Frame.__init__(self)
		self.master.title('Lane\'s MineSweeper')
		self.master.rowconfigure(0,weight=1)
		self.master.columnconfigure(0,weight=1)
		self.grid(sticky=N+S+E+W)

		self.canvas=Canvas(self,width=self.cols*SQ+1, height=self.rows*SQ+1, bg='white')
		self.canvas.grid(row=0,column=0)

		self.bind_all('<KeyPress>', self.keyboard)

		self.draw()

	def __init__(self):
		self.showing_mines = 0

		self.mines = MIN_MINES
		self.rows = GRID_R
		self.cols = GRID_C
		
		self.game = Minesweeper(self.rows, self.cols, self.mines)
		self.model = []
		self.model_selection = 0
		self.ensemble = [] # (model, weight)

		self.row = 0
		self.col = 0

		self.invent_canvas()

if __name__ == '__main__':
	if selection == TRAIN_FROM_FILE:
		train_model_from_file(training_file, build_resnet(32, (3,3), 3), (GRID_R, GRID_C))
	elif selection == BUILD:
		build_difficulty_stats( GRID_R, GRID_C, 10000 )
	elif selection == EVALUATE:
		evaluate(1000, load_model(trained_model))
	elif selection == EXPLORE:
		exploratory_model_training(build_resnet(64, (3,3), 5))
	elif selection == PLAY:
		display().mainloop()
	elif selection == RL:
		rl_training(build_resnet(64, (3,3), 5))
