#include <NeuralNetwork.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <future>
#include <thread>

using namespace std;
using namespace NN;

int main()
{
	unsigned int n = thread::hardware_concurrency();
	cout << n << " concurrent threads are supported.\n";

	typedef NeuralNetwork<float> float_network;
	typedef DataSet<float> float_dataset;
	typedef BatchTrainer<float_network, float_dataset> float_trainer;

	float_network ann({8, 16, 8, 1});

	float_dataset ds("xor8.txt", 256, 8, 1);
	float_trainer t(ann, ds);
	cout<<"Final training RMSE: "<<t.train(100)<<"\n";
	return 0;
}


// int main(int argc, const char* argv[])
// {

// 	std::cout<<"Testing.....\n";
// 	return 0;
// }
