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

	typedef float data_type;
	typedef NeuralNetwork<data_type> float_network;
	typedef DataSet<data_type> float_dataset;
	typedef MultithreadTrainer<float_network, float_dataset> multi_trainer;
	typedef Trainer<float_network, float_dataset> float_trainer;

	float_network ann_mt({8, 8, 4, 1});
	float_network ann_st(ann_mt);
	float_dataset ds(argv[1], 256, 8, 1);


	// float_network ann({2, 3, 1});
	// float_dataset ds(argv[1], 4, 2, 1);
	multi_trainer mt(ann_mt, ds, 4, 0.01f, 0.9f);

	float_trainer st(ann_st, ds, 0.01f, 0.9f);

	n = 10;
	while (n--){
		cout<<"mt: "<<mt.train(1)<<"\n";
		cout<<"st: "<<st.train_batch(1)<<"\n\n";
	}
	// data_type err = t.train_batch(1000000);
	// data_type err = t.train(1000000);
	// cout<<"Final training RMSE: "<<err<<"\n";
	return 0;
}


// int main(int argc, const char* argv[])
// {

// 	std::cout<<"Testing.....\n";
// 	return 0;
// }
