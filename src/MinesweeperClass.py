
import os

import numpy as np
from numpy.random import random as npr

class Minesweeper():

	def __init__(self, rows, cols, number_of_mines):
		self.DEFEAT = -1
		self.INPROGRESS = 0
		self.VICTORY = 1
		
		self._first_move_has_been_made = False
		self.game_status = self.INPROGRESS

		self.rows = rows
		self.cols = cols

		self.mine_field = self.initialize_mine_field( number_of_mines )
		self.proximity_field = self.initialize_proximity_field( self.mine_field )
		self.visible_cells = self.initialize_visible_cells( self.mine_field )
		self.flagged_cells = self.initialize_flagged_cells( self.mine_field )
		self.border_cells = self.initialize_border_cells( self.mine_field )

	def reset(self):
		self.visible_cells = self.initialize_visible_cells( self.mine_field )
		self.flagged_cells = self.initialize_flagged_cells( self.mine_field )
		self.update_border_cells()
		self.game_status = self.INPROGRESS

	def is_visible(self, move):
		r = move[0]
		c = move[1]
		return self.visible_cells[r][c]

	def forfeit_game(self):
		self.game_status = self.DEFEAT

	def first_move_has_been_made(self):
		return self._first_move_has_been_made

	def this_cell_is_mined(self, move):
		r = move[0]
		c = move[1]
		if self.mine_field[r][c]:
			return True
		else:
			return False

	def this_cell_is_flagged(self, move):
		r = move[0]
		c = move[1]
		if self.flagged_cells[r][c]:
			return True
		else:
			return False

	def get_field_shape(self):
		return self.mine_field.shape
	
	def get_game_status(self):
		return self.game_status

	def get_moves(self):
		unvisited_cells = []
		for r in range(self.rows):
			for c in range(self.cols):
				# if you don't step on flagged cells, you won't learn to avoid them
				if self.visible_cells[r][c] == 0:# and self.flagged_cells[r][c] == 0:
					unvisited_cells.append((r,c))
		return unvisited_cells
	
	def get_moves_for_rl(self):
		unvisited_cells = []
		for r in range(self.rows):
			for c in range(self.cols):
				unvisited_cells.append(self.visible_cells[r][c]^1)
		return unvisited_cells
	
	def get_border_cells(self):
		bordering_cells = []
		for r in range(self.rows):
			for c in range(self.cols):
				if self.border_cells[r][c] == 1:# and self.flagged_cells[r][c] == 0:
					bordering_cells.append((r,c))

		if len(bordering_cells) == 0:
			return self.get_moves()
		else:
			return bordering_cells
	
	def get_mine_field(self):
		return self.mine_field
	
	def get_proximity_field(self):
		return self.proximity_field
	
	def get_visible_field(self):
		return self.visible_cells
	
	def get_flagged_field(self):
		return self.flagged_cells

	def get_border_field(self):
		return self.border_cells

	def point_is_on_board(self, r, c):
		if r >= 0 and r < self.rows and c >= 0 and c < self.cols:
			return True
		return False

	def generate_fresh_field(self, int_size, shape_tuple):
		return np.zeros(int_size,dtype=np.intc).reshape(shape_tuple)

	# generate a new field and plant mines in it
	def initialize_mine_field(self, number_of_mines):
		
		new_field = self.generate_fresh_field(self.rows*self.cols, (self.rows,self.cols))
		
		number_of_cells = self.rows * self.cols
		if number_of_cells < number_of_mines:
			print ('The number of mines',number_of_mines,'is greater than the number of cells',number_of_cells)
			return new_field

		for n in range(number_of_mines):
			r = int(npr()*self.rows)
			c = int(npr()*self.cols)
			
			# if there's already a mine there, then try somewhere else
			while new_field[r][c]:
				r = int(npr()*self.rows)
				c = int(npr()*self.cols)
				
			new_field[r][c] = 1
		
		return new_field
	
	# generate a field saying how many mines are adjacent to this cell
	def initialize_proximity_field(self, minefield):
		new_field = self.generate_fresh_field(minefield.size, minefield.shape)

		# visit each cell
		for R in range(self.rows):
			for C in range(self.cols):
				mines = 0
				# checking the environs for a mine
				for r in range(R-1,R+2):
					for c in range(C-1,C+2):
						if self.point_is_on_board(r, c):
							if minefield[r][c]:
								mines += 1
				new_field[R][C] = mines

		return new_field
	
	# generate a field saying how many mines are adjacent to this cell
	def update_border_cells(self):
		self.border_cells = self.generate_fresh_field(self.mine_field.size, self.mine_field.shape)

		# visit each cell
		for R in range(self.rows):
			for C in range(self.cols):
				# if the current cell is unvisited, then check neighbors to see if I'm a border cell
				if self.visible_cells[R][C] == 0:
					# checking the environs for a mine
					for r in range(R-1,R+2):
						for c in range(C-1,C+2):
							if self.point_is_on_board(r, c):
								if self.visible_cells[r][c]:
									self.border_cells[R][C] = 1

	def initialize_border_cells(self, minefield):
		return self.generate_fresh_field(minefield.size, minefield.shape)

	# generate a field where, by default, nothing is visible
	def initialize_visible_cells(self, minefield):
		return self.generate_fresh_field(minefield.size, minefield.shape)

	# generate a field where, by default, no cells are flagged
	def initialize_flagged_cells(self, minefield):
		return self.generate_fresh_field(minefield.size, minefield.shape)

	def set_flag(self, move, value):
		r = move[0]
		c = move[1]
		if self.point_is_on_board(r, c):
			# you're only able to set or remove flags on cells that are not visible
			if self.visible_cells[r][c] == 0:
				self.flagged_cells[r][c] = value

	def place_flag(self, move):
		self.set_flag( move, 1 )

	def remove_flag(self, move):
		self.set_flag( move, 0 )
	
	def set_mine(self, move, value):
		r = move[0]
		c = move[1]
		if self.point_is_on_board(r, c):
			self.mine_field[r][c] = value
		self.proximity_field = self.initialize_proximity_field(self.mine_field)

	def set_visible(self, move):
		r = move[0]
		c = move[1]
		if self.point_is_on_board(r, c):
			self.visible_cells[r][c] = 1
		self.update_border_cells()

	def set_invisible(self, move):
		r = move[0]
		c = move[1]
		if self.point_is_on_board(r, c):
			self.visible_cells[r][c] = 0
		self.update_border_cells()

	def place_mine(self, move):
		self.set_invisible( move )
		self.set_mine( move, 1 )
		self.revive_game()

	def revive_game(self):
		self.game_status = self.INPROGRESS

	def remove_mine(self, move):
		self.set_mine( move, 0 )
	
	def visit_cell(self, move):
		self._first_move_has_been_made = True

		r = move[0]
		c = move[1]
		if self.point_is_on_board(r, c):
			# you can't visit a cell that is currently flagged
			if self.flagged_cells[r][c] == 0:
				self.visible_cells[r][c] = 1
				self.update_cells()
				self.update_border_cells()

	def render_unto_visible_what_is_visible(self):
		still_updating = True

		while still_updating:
			still_updating = False
			'''
			1) visit each cell
			2) if the current cell is not visible, then
				3) if any adjacent cell is visible and has zero adjacent mines, then
					4) set the current cell to visible
					5) and set still_updating = True
			'''
			# visit each cell
			for R in range(self.rows):
				for C in range(self.cols):
					# if this cell is not visible
					if self.visible_cells[R][C] == 0:
						# then check environs for 'safe' cells that are visible
						for r in range(R-1,R+2):
							for c in range(C-1,C+2):
								if self.point_is_on_board(r, c):
									if self.visible_cells[r][c] != 0 and self.proximity_field[r][c] == 0:
										self.visible_cells[R][C] = 1
										still_updating = True
			
	def update_game_status(self):
		i_stepped_on_a_mine = False
		i_won = True

		for r in range(self.rows):
			for c in range(self.cols):
				if self.visible_cells[r][c] == self.mine_field[r][c]:
					# if a mine is visible, it's because we stepped on it.  oops!
					if self.mine_field[r][c] != 0:
						i_stepped_on_a_mine = True
					# if a cell with no mines is invisible, then we haven't finished sweeping
					if self.mine_field[r][c] == 0:
						i_won = False

		self.game_status = self.INPROGRESS
		if i_stepped_on_a_mine:
			self.game_status = self.DEFEAT
		elif i_won:
			self.game_status = self.VICTORY

	def update_cells(self):
		self.render_unto_visible_what_is_visible()
		self.update_game_status()

		return self.game_status
	
	def report(self):
		pass
		'''
		print self.mine_field
		print
		print self.proximity_field
		print
		print self.visible_cells
		print
		print self.flagged_cells
		print
		print self.border_cells
		'''
