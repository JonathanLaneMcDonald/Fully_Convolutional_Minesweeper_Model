
// I originally wrote this in linux. Posix threads don't seem to work properly in Windows. I'll need to look into that.
// For now, run with one thread
// Usage: minesweeper [rows] [cols] [mines] [samples] [threads] > outfile
// Note: train.py is expecting 'training' and 'validation' files, so use one or the other in place of "outfile" in the line above
// What a janky way to do comments, lol. I really need to update this stuff with proper options :D

// g++ minesweeper_main.cpp -o minesweeper -lpthread -O3

#include "minesweeper.h"

#include <mutex>
#include <thread>

/*
Play a specified number of games of minesweeper across a specified number of threads.
Select a random frame from each game played and write it to stdout.
This output can be redirected to a file and used for training by calling train.py without modification.
*/
void play_games(int id, int rows, int cols, int mines, int samples, std::mutex* mutex)
{
	std::random_device rd;
    std::mt19937 _gen(rd());
    std::uniform_real_distribution<> _dis(0.0, 1.0);

	MineSweeper game = MineSweeper(rows, cols, mines, _gen, _dis);

	for (int i = 0; i < samples; i++)
	{
		while (game.status() == GameState::Underway || game.status() == GameState::Fresh)
			game.make_random_safe_border_move();

		auto frames = game.get_frames();
		int index = int(frames.size() * _dis(_gen));

		mutex->lock();
		std::cout << frames[index] << std::endl;
		mutex->unlock();

		game.reset();
	}

	return;
}

int main(int argc, char* argv[])
{
	std::mutex mutex;
	std::vector<std::thread> workers(0);

	int rows = std::stoi(argv[1]);
	int cols = std::stoi(argv[2]);
	int mines = std::stoi(argv[3]);
	int samples = std::stoi(argv[4]);
	int threads = std::stoi(argv[5]);
	
	for (int i = 0; i < threads; i++)
	{
		std::thread worker(play_games, i, rows, cols, mines, int(samples/threads), &mutex);
		workers.push_back(std::move(worker));
	}

	for (int i = 0; i < workers.size(); i++)
		workers[i].join();

	return 0;
}


