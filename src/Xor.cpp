#include <NN.hpp>

#include <iostream>
#include <cassert>
#include <stdlib.h>

using namespace std;
using namespace NN;

void usage(){
    cout<< "app thread_num\n";
}

int main(int argc, const char* argv[])
{
	//(void) argc, (void)argv;
	// assert("Must provide file name" && argc > 1);
    if ( 2 != argc ){
        usage();
        exit(1);
    }
    size_t thread_num = 4; // default is 4 threads
    thread_num = static_cast<size_t>(atoi(argv[1]));   
    if (0==thread_num){
        thread_num = 4;
    }
    cout<< "Thread number has been set as "<<thread_num<<"\n";
	unsigned int n = thread::hardware_concurrency();
	cout << n << " concurrent threads are supported.\n";

	typedef float data_type;
	typedef NeuralNetwork<data_type> float_network;
	typedef DataSet<data_type> float_dataset;
	typedef MultithreadTrainer<float_network, float_dataset> multi_trainer;
	// typedef Trainer<float_network, float_dataset> float_trainer;

	float_network ann({8, 4, 2, 1}, ActFn::Tanh, ActFn::Linear);
	float_dataset ds_train("data/xor8.txt", 256, 8, 1);	
	multi_trainer mt(ann, ds_train, thread_num);
	cout<<"mt.train(50000, 50) will be executed\n";
	mt.train(50000, 50);
	return 0;
}
