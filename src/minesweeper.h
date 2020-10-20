
#include <string>
#include <random>
#include <iostream>
#include <vector>
#include <tuple>

enum class GameState
{
	Fresh = 0,
	Underway,
	Victory,
	Defeat
};

enum class CellState
{
	// also storing proximity in the board, so these flags can start later
	Visible = 10,
	Border,
	Mined
};

/*
* I want to be able to do bitwise operations with CellStates, so overload this operator
*/
int operator<<(const int &a, const CellState &b)
{
	return a<<static_cast<int>(b);
}

/*
* MineSweeper implements the most basic aspects of the game to build a dataset for training a model.
* Safe moves are identified at each time step and a random safe move is selected to advance the game.
*/
class MineSweeper
{
public:

	/*
	* Construct an instance of MineSweeper and initialize it.
	* 
	* @param	rows		The number of rows the minefield is to have
	* @param	cols		The number of columns the minefield will have
	* @param	mines		The number of mines
	* @param	gen			The entropy source -- a Mersenne Twister in this case
	* @param	dis			The distribution being sampled by the entropy source
	* 
	* Note: We're interested in entropy and uniform distributions here so we can (pseudo)randomly place mines and make moves
	*/
	MineSweeper(int rows, int cols, int mines, std::mt19937& gen, std::uniform_real_distribution<>& dis)
	: 	_rows(rows),
		_cols(cols),
		_mines(mines),
		_gen(gen),
		_dis(dis)
	{
		reset();
	}

	/*
	* (Re)Initialize the minefield.
	*/
	void reset()
	{
		_status = GameState::Fresh;

		_frames.resize(0);

		_board.resize(0, std::vector<int>(0));
		_board.resize(_rows, std::vector<int>(_cols));

		initialize_mine_field();
		initialize_proximity_field();
		update_border_field();
	}

	/*
	* Distribute mines across the minefield.
	*/
	void initialize_mine_field()
	{
	    int r, c;
		for (int placed_mines = 0; placed_mines < _mines; )
		{
			r = int(_rows * _dis(_gen));
			c = int(_cols * _dis(_gen));
			if (!(_board[r][c] & (1<<CellState::Mined)))
			{
				_board[r][c] |= (1<<CellState::Mined);
				placed_mines ++;
			}
		}
	}

	/*
	* Build the field that shows how many mines neighbor each cell.
	*/
	void initialize_proximity_field()
	{
		for (int row = 0; row < _rows; row ++)
			for (int col = 0; col < _cols; col ++)
			{
				int mined_neighbors = 0;
				for (int r = row-1; r < row+2; r ++)
					for (int c = col-1; c < col+2; c ++)
						if (0 <= r && r < _rows && 0 <= c && c < _cols && _board[r][c]&(1<<CellState::Mined))
							mined_neighbors ++;
				_board[row][col] |= mined_neighbors;
			}
	}

	/*
	* Identify cells that border visible ones because these are important for training a model.
	*/
	void update_border_field()
	{
		for (int row = 0; row < _rows; row ++)
			for (int col = 0; col < _cols; col ++)
			{
				int visible_neighbors = false;
				for (int r = row-1; r < row+2; r ++)
					for (int c = col-1; c < col+2; c ++)
						if (0 <= r && r < _rows && 0 <= c && c < _cols && _board[r][c]&(1<<CellState::Visible))
							visible_neighbors = true;
				visible_neighbors && !(_board[row][col] & (1<<CellState::Visible)) ? _board[row][col] |= (1<<CellState::Border) : _board[row][col] &= ~(1<<CellState::Border);
			}
	}

	/*
	* Visit a cell and handle the implications.
	* 
	* @param	this_move		A tuple of ints specifying the <row, col> of this_move (should probably make a struct for this)
	*/
	void visit_cell(std::tuple<int, int> this_move)
	{
		std::vector<std::tuple<int, int> > moves(0);
		moves.push_back(this_move);

		while (!moves.empty())
		{
			auto move = moves.back();
			moves.pop_back();

			int row = std::get<0>(move);
			int col = std::get<1>(move);

			_board[row][col] |= (1<<CellState::Visible);

			//if i know there are no mines nearby, then i'll reveal and visit everything near me
			if (!(_board[row][col]&1023))
				for (int r = row-1; r < row+2; r++)
					for (int c = col-1; c < col+2; c++)
						if (0 <= r && r < _rows && 0 <= c && c < _cols)
						{
							if (!(_board[r][c]&1023) && !(_board[r][c]&(1<<CellState::Visible)))
								moves.push_back(std::make_tuple(r,c));

							_board[r][c] |= (1<<CellState::Visible);
						}

		}

		update_border_field();

		state_to_frame();

		_status = GameState::Underway;
	}

	/*
	* Randomly make a safe move - we prefer making safe border moves, but if one isn't available, this will do.
	*/
	void make_random_safe_move()
	{

		std::vector<std::tuple<int, int> > moves(0);
		for (int r = 0; r < _rows; r++)
			for (int c = 0; c < _cols; c++ )
				if (!(_board[r][c] & (1<<CellState::Visible)) && !(_board[r][c] & (1<<CellState::Mined)))
				{
					moves.push_back(std::make_tuple(r,c));
				}

		if (moves.empty())
		{
			_status = GameState::Victory;
		}
		else
		{
			auto move = moves[int(moves.size() * _dis(_gen))];
			
			visit_cell(move);
		}
	}

	/*
	* Make a random safe border move.
	* 
	* Note: This matters because we're using this to train a model, so we'll prefer to make moves that have information about them
	* that will be available for a model to learn.
	*/
	void make_random_safe_border_move()
	{

		std::vector<std::tuple<int, int> > moves(0);
		for (int r = 0; r < _rows; r++)
			for (int c = 0; c < _cols; c++ )
				if (!(_board[r][c] & (1<<CellState::Visible)) && !(_board[r][c] & (1<<CellState::Mined)) && (_board[r][c] & (1<<CellState::Border)))
				{
					moves.push_back(std::make_tuple(r,c));
				}

		if (moves.empty())
		{
			make_random_safe_move();
		}
		else
		{
			auto move = moves[int(moves.size() * _dis(_gen))];

			visit_cell(move);
		}
	}

	/*
	* Write the current board state, including known safe cells, as a string
	*/
	void state_to_frame()
	{
		char message[_rows*_cols+1];
		message[_rows*_cols] = 0;

		for (int r = 0; r < _rows; r++)
			for (int c = 0; c < _cols; c++)
			{
				if (_board[r][c]&(1<<CellState::Visible))
					message[r*_cols + c] = char(48 + _board[r][c]&1023);
				else if(_board[r][c]&(1<<CellState::Mined))
					message[r*_cols + c] = char(33);
				else if(_board[r][c]&(1<<CellState::Border))
					message[r*_cols + c] = char(115);
				else
					message[r*_cols + c] = char(63);
			}

		_frames.push_back(std::string(message));
		//std::cout << _frames.back() << std::endl;
	}

	/*
	* Return a const ref to the current frames stack
	*/
	const std::vector<std::string>& get_frames()
	{
		return _frames;
	}

	/*
	* Get a visual printout of everything that's happening right now
	*/
	void print_everything()
	{
		print_board("visibility", 1<<CellState::Visible);
		print_board("border cells", 1<<CellState::Border);
		print_board("mines", 1<<CellState::Mined);
		print_board("proximity", 1023, false);
	}

	/*
	* Get a visual representation of a particular aspect of the game
	*/
	void print_board(const std::string message, int mask, bool mask_is_bitwise=true)
	{
		std::cout << message << std::endl << "board size=" << _board.size() << "x" << _board[0].size() << std::endl;
		for (int i = 0; i < _board.size(); i++ )
		{
			for (int j = 0; j < _board[i].size(); j++)
			{
				if (mask_is_bitwise)
					std::cout << ((_board[i][j] & mask) ? 1 : 0) << " ";
				else
					std::cout << (_board[i][j] & mask) << " ";
			}
			std::cout << std::endl;
		}		
	}

	GameState _status;

private:
	int _rows;
	int _cols;
	int _mines;

	std::mt19937& _gen;
	std::uniform_real_distribution<>& _dis;

	std::vector<std::string> _frames;
	std::vector<std::vector<int> > _board;
};
