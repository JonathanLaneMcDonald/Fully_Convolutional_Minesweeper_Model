
//g++ main.cpp -o minesweeper -lpthread -std=c++11 -O2

#include <thread>
#include <mutex>

#include "minesweeper.h"

void play_games(int id, int rows, int cols, int mines, int samples, std::mutex* mutex)
{
	std::random_device rd;
    std::mt19937 _gen(rd());
    std::uniform_real_distribution<> _dis(0.0, 1.0);

	MineSweeper game = MineSweeper(rows,cols,mines,_gen,_dis);

	for (int i = 0; i < samples; i++)
	{
		//game.print_everything();
		while (game._status == UNDERWAY || game._status == FRESH)
			game.make_random_safe_border_move();

		auto frames = game.get_frames();
		int index = int(frames.size() * _dis(_gen));

		mutex->lock();
		std::cout << frames[index] << std::endl;
		mutex->unlock();

		//game.print_everything();

		game.reset();
	}

	return;
}

int main()
{
	std::mutex mutex;
	std::vector<std::thread> workers(0);

	int rows = 16;
	int cols = 16;
	int mines = 40;
	int samples = 1<<20;
	int threads = 4;
	
	for (int i = 0; i < threads; i++)
	{
		std::thread worker(play_games, i, rows, cols, mines, int(samples/threads), &mutex);
		workers.push_back(std::move(worker));
	}

	for (int i = 0; i < workers.size(); i++)
		workers[i].join();

	return 0;
}


