
from common import *
from evaluate import generate_heat_map, select_top_moves
from MinesweeperClass import *

from keras.models import load_model

from tkinter import *

SQ = 24


def itoh(n):
	h = '0123456789abcdef'
	return h[(n >> 20) & 0xf] + h[(n >> 16) & 0xf] + h[(n >> 12) & 0xf] + h[(n >> 8) & 0xf] + h[(n >> 4) & 0xf] + h[(n >> 0) & 0xf]


class MinesweeperUI(Frame):

	def draw(self):
		self.canvas.delete('all')
		
		prediction = None
		if self.model and self.use_model:
			prediction = generate_heat_map(self.game, self.model)

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
					self.canvas.create_rectangle(1+c*SQ, 1+r*SQ, 1+(c+1)*SQ, 1+(r+1)*SQ, fill='yellow', outline=outline)
					fill = 'yellow'
				elif visi_field[r][c]:
					if mine_field[r][c]:
						self.canvas.create_rectangle(1+c*SQ, 1+r*SQ, 1+(c+1)*SQ, 1+(r+1)*SQ, fill='red', outline=outline)
					else:
						self.canvas.create_rectangle(1+c*SQ, 1+r*SQ, 1+(c+1)*SQ, 1+(r+1)*SQ, fill='white', outline=outline)
						if prox_field[r][c]:
							self.canvas.create_text(c*SQ+SQ/2, r*SQ+SQ/2, font=font, text=str(prox_field[r][c]))
				else:
					self.canvas.create_rectangle(1+c*SQ, 1+r*SQ, 1+(c+1)*SQ, 1+(r+1)*SQ, fill='grey', outline=outline)
					fill = 'grey'

				# these cells are 'border cells'
				if bord_field[r][c]:
					self.canvas.create_rectangle(2+c*SQ, 2+r*SQ, (c+1)*SQ, (r+1)*SQ, fill=fill, outline='yellow')

				# this is where the 'cursor' is
				if r == self.row and c == self.col:
					self.canvas.create_rectangle(2+c*SQ, 2+r*SQ, (c+1)*SQ, (r+1)*SQ, fill=fill, outline='red')
					if visi_field[r][c]:
						if mine_field[r][c]:
							self.canvas.create_rectangle(1+c*SQ, 1+r*SQ, 1+(c+1)*SQ, 1+(r+1)*SQ, fill='red', outline=outline)
						elif prox_field[r][c]:
							self.canvas.create_text(c*SQ+SQ/2, r*SQ+SQ/2, font=font, text=str(prox_field[r][c]))

		if prediction:
			# find the minimum and maximum extents of the prediction values - this way, we can assign percentage values to cells
			sum_over_interesting = []
			for p in prediction:
				r = p[1][0]
				c = p[1][1]
				
				if visi_field[r][c] == 0 and flag_field[r][c] == 0:
					sum_over_interesting.append(p[0])

			if len(sum_over_interesting):
				coldest = min(sum_over_interesting)
				warmest = max(sum_over_interesting)
				t_range = warmest - coldest

				visualize = []
				if t_range:
					for p in prediction:
						r = p[1][0]
						c = p[1][1]

						value = 0
						if coldest <= p[0] <= warmest:
							value = float(p[0] - coldest) / t_range

						visualize.append((value, (r, c)))

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

					red = max(0, red)
					green = max(0, green)
					blue = max(0, blue)

					color = (red << 16) + (green << 8) + blue

					r = v[1][0]
					c = v[1][1]

					if visi_field[r][c]:
						self.canvas.create_oval(3+c*SQ, 3+r*SQ, c*SQ+SQ/3, r*SQ+SQ/3, fill='white')
					else:
						self.canvas.create_oval(3+c*SQ, 3+r*SQ, c*SQ+SQ/3, r*SQ+SQ/3, fill='#'+itoh(color))

					if r == self.row and c == self.col:
						print('Model Predicts:', value)

		if self.showing_mines:
			for r in range(game_rows):
				for c in range(game_cols):
					if mine_field[r][c]:
						self.canvas.create_oval(c*SQ+SQ/4, r*SQ+SQ/4, c*SQ+3*SQ/4, r*SQ+3*SQ/4, fill='red')

		self.canvas.update_idletasks()

	def nnsolver(self):
		if self.use_model:
			while self.game.get_game_status() == self.game.INPROGRESS:
				safe_moves = [x for x in self.game.get_moves() if not self.game.this_cell_is_mined(x)]

				if len(safe_moves):
					if self.game.first_move_has_been_made():
						selected_move = select_top_moves(self.game, self.model, 1)[0]
						if self.game.this_cell_is_mined(selected_move):
							self.game.forfeit_game()
						else:
							self.game.place_flag(selected_move)
							self.draw()
							self.game.remove_flag(selected_move)

						self.game.visit_cell(selected_move)
					else:
						generous_first_moves = self.game.get_generous_first_moves()
						if len(generous_first_moves):
							self.game.visit_cell(generous_first_moves[int(npr()*len(generous_first_moves))])
						else:
							self.game.visit_cell(safe_moves[int(npr()*len(safe_moves))])
				else:
					self.game.forfeit_game()

				self.draw()

	def keyboard(self, event):
		if event.keysym == 'Escape':
			self.quit()

		if event.keysym == 'Up':
			self.row -= 1
		if event.keysym == 'Down':
			self.row += 1
		if event.keysym == 'Left':
			self.col -= 1
		if event.keysym == 'Right':
			self.col += 1

		if self.row < 0:
			self.row = 0
		if self.row >= self.rows:
			self.row = self.rows-1
		if self.col < 0:
			self.col = 0
		if self.col >= self.cols:
			self.col = self.cols-1

		if event.char == 'e':
			self.mines += 1
		if event.char == 'q':
			self.mines -= 1
		if event.char == 'e' or event.char == 'q':
			print('Mines:', self.mines)
		
		if event.char == 'i':
			self.use_model ^= 1

		if event.keysym == 'g':
			self.game = Minesweeper(self.rows, self.cols, self.mines)

		if event.keysym == 'm':
			self.showing_mines ^= 1

		message = ''
		if self.game:
			if event.keysym == 's':
				self.nnsolver()

			if event.keysym == 'r':
				self.game.reset()

			if event.keysym == 'space':
				if self.game.is_visible((self.row, self.col)):
					self.game.set_invisible((self.row, self.col))
				else:
					message = 'game.visit_cell('+str(self.row)+','+str(self.col)+')'
					self.game.visit_cell((self.row, self.col))

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
					print('yay, victory!')
			elif self.game.game_status == self.game.DEFEAT:
				for i in range(10):
					print('boo, you lose')

		print(event.keysym, event.char, message)

		self.draw()
	
	def __init__(self):
		Frame.__init__(self)
		self.master.title('Lane\'s MineSweeper')
		self.master.rowconfigure(0, weight=1)
		self.master.columnconfigure(0, weight=1)
		self.grid(sticky=N+S+E+W)

		self.showing_mines = 0

		self.mines = MIN_MINES
		self.rows = GRID_R
		self.cols = GRID_C
		
		self.canvas = Canvas(self, width=self.cols*SQ+1, height=self.rows*SQ+1, bg='white')
		self.canvas.grid(row=0, column=0)

		self.game = Minesweeper(self.rows, self.cols, self.mines)
		self.model = load_model('minesweeper model 8x8x10')
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.use_model = 0

		self.row = 0
		self.col = 0

		self.bind_all('<KeyPress>', self.keyboard)

		self.draw()


MinesweeperUI().mainloop()
