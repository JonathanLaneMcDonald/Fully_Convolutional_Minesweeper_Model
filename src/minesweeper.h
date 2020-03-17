

#include <string>
#include <random>
#include <iostream>
#include <vector>
#include <tuple>

enum
{
	FRESH = 0,
	UNDERWAY,
	VICTORY,
	DEFEAT
};

enum
{
	// also storing proximity in the board, so these flags can start later
	VISIBLE = 10,
	BORDER,
	MINED
};

class MineSweeper
{
public:
	MineSweeper(const int rows, const int cols, const int mines, std::mt19937& gen, std::uniform_real_distribution<>& dis)
	: 	_rows(rows),
		_cols(cols),
		_mines(mines),
		_gen(gen),
		_dis(dis)
	{
		reset();
	}

	void reset()
	{
		_status = FRESH;

		_frames.resize(0);

		_board.resize(0, std::vector<int>(0));
		_board.resize(_rows, std::vector<int>(_cols));

		initialize_mine_field();
		initialize_proximity_field();
		update_border_field();
	}

	void initialize_mine_field()
	{
	    int r, c;
		for (int placed_mines = 0; placed_mines < _mines; )
		{
			r = int(_rows * _dis(_gen));
			c = int(_cols * _dis(_gen));
			if (!(_board[r][c] & (1<<MINED)))
			{
				_board[r][c] |= (1<<MINED);
				placed_mines ++;
			}
		}
	}

	void initialize_proximity_field()
	{
		for (int row = 0; row < _rows; row ++)
			for (int col = 0; col < _cols; col ++)
			{
				int mined_neighbors = 0;
				for (int r = row-1; r < row+2; r ++)
					for (int c = col-1; c < col+2; c ++)
						if (0 <= r && r < _rows && 0 <= c && c < _cols && _board[r][c]&(1<<MINED))
							mined_neighbors ++;
				_board[row][col] |= mined_neighbors;
			}
	}

	void update_border_field()
	{
		for (int row = 0; row < _rows; row ++)
			for (int col = 0; col < _cols; col ++)
			{
				int visible_neighbors = false;
				for (int r = row-1; r < row+2; r ++)
					for (int c = col-1; c < col+2; c ++)
						if (0 <= r && r < _rows && 0 <= c && c < _cols && _board[r][c]&(1<<VISIBLE))
							visible_neighbors = true;
				visible_neighbors && !(_board[row][col] & (1<<VISIBLE)) ? _board[row][col] |= (1<<BORDER) : _board[row][col] &= ~(1<<BORDER);
			}
	}

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

			_board[row][col] |= (1<<VISIBLE);

			//if i know there are no mines nearby, then i'll reveal and visit everything near me
			if (!(_board[row][col]&1023))
				for (int r = row-1; r < row+2; r++)
					for (int c = col-1; c < col+2; c++)
						if (0 <= r && r < _rows && 0 <= c && c < _cols)
						{
							if (!(_board[r][c]&1023) && !(_board[r][c]&(1<<VISIBLE)))
								moves.push_back(std::make_tuple(r,c));

							_board[r][c] |= (1<<VISIBLE);
						}

		}

		update_border_field();

		state_to_frame();

		_status = UNDERWAY;
	}

	void make_random_safe_move()
	{

		std::vector<std::tuple<int, int> > moves(0);
		for (int r = 0; r < _rows; r++)
			for (int c = 0; c < _cols; c++ )
				if (!(_board[r][c] & (1<<VISIBLE)) && !(_board[r][c] & (1<<MINED)))
				{
					moves.push_back(std::make_tuple(r,c));
				}

		if (moves.empty())
		{
			_status = VICTORY;
		}
		else
		{
			auto move = moves[int(moves.size() * _dis(_gen))];
			
			visit_cell(move);
		}
	}

	void make_random_safe_border_move()
	{

		std::vector<std::tuple<int, int> > moves(0);
		for (int r = 0; r < _rows; r++)
			for (int c = 0; c < _cols; c++ )
				if (!(_board[r][c] & (1<<VISIBLE)) && !(_board[r][c] & (1<<MINED)) && (_board[r][c] & (1<<BORDER)))
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

	void state_to_frame()
	{
		char message[_rows*_cols+1];
		message[_rows*_cols] = 0;

		for (int r = 0; r < _rows; r++)
			for (int c = 0; c < _cols; c++)
			{
				if (_board[r][c]&(1<<VISIBLE))
					message[r*_cols + c] = char(48 + _board[r][c]&1023);
				else if(_board[r][c]&(1<<MINED))
					message[r*_cols + c] = char(33);
				else if(_board[r][c]&(1<<BORDER))
					message[r*_cols + c] = char(115);
				else
					message[r*_cols + c] = char(63);
			}

		_frames.push_back(std::string(message));
		//std::cout << _frames.back() << std::endl;
	}

	const std::vector<std::string>& get_frames()
	{
		return _frames;
	}

	void print_everything()
	{
		print_board("visibility", 1<<VISIBLE);
		print_board("border cells", 1<<BORDER);
		print_board("mines", 1<<MINED);
		print_board("proximity", 1023, false);
	}

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

	int _status;

private:
	int _rows;
	int _cols;
	int _mines;

	std::mt19937& _gen;
	std::uniform_real_distribution<>& _dis;

	std::vector<std::string> _frames;
	std::vector<std::vector<int> > _board;
};
