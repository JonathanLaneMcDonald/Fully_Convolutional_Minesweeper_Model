
"""
Board Sizes:
	beginner 		8x8 with 10 mines		15.625% mines
	intermediate	16x16 with 40 mines		15.625% mines
	expert			16x30 with 99 mines		~20% mines
	
	easy			16x32 with 64 mines		12.50% mines
	medium			16x32 with 96 mines		18.75% mines
	hard			16x32 with 128 mines	25.00% mines
"""

CHANNELS = 10

GRID_R = 16
GRID_C = 30
GRID_CELLS = GRID_R * GRID_C

MIN_MINES = 99
MAX_MINES = MIN_MINES
