#include <NeuralNetwork.hpp>

#include <iostream>
// #include <vector>
// #include <algorithm>
// #include <numeric>
// #include <future>
#include <thread>

#include <cassert>

using namespace std;
using namespace NN;

int main(int argc, const char* argv[])
{
	assert("Must provide file name" && argc > 1);
	unsigned int n = thread::hardware_concurrency();
	cout << n << " concurrent threads are supported.\n";

	typedef NeuralNetwork<float> float_network;
	typedef DataSet<float> float_dataset;
	typedef BatchTrainer<float_network, float_dataset> float_trainer;

	float_network ann({8, 16, 16, 1});
	float_dataset ds(argv[1], 256, 8, 1);
	float_trainer t(ann, ds, 0.05, 0.9);
	float err = t.train(10000);
	cout<<"Final training RMSE: "<<err<<"\n";
	return 0;
}


// int main(int argc, const char* argv[])
// {

// 	std::cout<<"Testing.....\n";
// 	return 0;
// }
