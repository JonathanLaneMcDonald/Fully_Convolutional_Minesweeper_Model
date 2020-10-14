from common import *
from MinesweeperClass import *

import numpy as np
from math import log

from keras.models import load_model


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


def generate_heat_map(game, model):
	model_prediction = model.predict(np.array(board_to_features(game)).reshape(1, GRID_R, GRID_C, CHANNELS))

	prediction = model_prediction.reshape(GRID_CELLS)

	flag_field = game.get_flagged_field()

	heat_map = []
	for i in range(len(prediction)):
		r = i // GRID_C
		c = i % GRID_C
		if flag_field[r][c]:
			# FIXME: don't change this value here.  check for flags somewhere else because this compromises the goal of the function
			heat_map.append((float(-1), (r, c)))
		else:
			heat_map.append((float(prediction[i]), (r, c)))

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
			top_moves.append((r, c))
			if len(top_moves) == moves_requested:
				return top_moves
	return top_moves


def build_difficulty_stats(rows, cols, games):
	for mines in range(MIN_MINES, MAX_MINES + 1):
		wins = 0
		losses = 0
		for i in range(games):
			game = Minesweeper(rows, cols, mines)

			while game.get_game_status() == game.INPROGRESS:
				unvisited_cells = game.get_moves()
				selection = int(npr() * len(unvisited_cells))

				r = unvisited_cells[selection][0]
				c = unvisited_cells[selection][1]

				game.visit_cell((r, c))

			if game.get_game_status() == game.VICTORY:
				wins += 1
			elif game.get_game_status() == game.DEFEAT:
				losses += 1

		print(mines, 'mines', wins, 'wins', losses, 'losses')


def evaluate(target_games, model, mines=MIN_MINES, nn_predicts_opening_move=False):
	deaths = 0
	moves_played = 0
	games_played = 0
	games_won = 0
	global_likelihood = 0
	while games_played < target_games:
		game = Minesweeper(GRID_R, GRID_C, mines)
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
					generous_first_moves = game.get_generous_first_moves()
					if len(generous_first_moves):
						game.visit_cell(generous_first_moves[int(npr() * len(generous_first_moves))])
					else:
						game.visit_cell(safe_moves[int(npr() * len(safe_moves))])
			else:
				game.forfeit_game()  # because we're surrounded by mines ;(

		global_likelihood += log(likelihood) / log(10)
		if not to_smitherines:
			games_won += 1

		if games_played % 1 == 0:
			print(
				games_played, float(deaths) / moves_played, float(games_won) / games_played,
				float(moves_played) / games_played, float(global_likelihood) / games_played)

	return float(deaths) / moves_played, float(games_won) / games_played, float(moves_played) / games_played, float(
		global_likelihood) / games_played


if __name__ == "__main__":
	model = load_model('minesweeper model 8x8x10')
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	evaluate(10000, model)
