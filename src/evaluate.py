
from common import *
from MinesweeperClass import *

import numpy as np

from keras.models import load_model

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


