#include <NN.hpp>

#include <iostream>
#include <cassert>

using namespace std;
using namespace NN;

int main(int argc, const char* argv[])
{
	(void) argc, (void)argv;
	// assert("Must provide file name" && argc > 1);
	unsigned int n = thread::hardware_concurrency();
	cout << n << " concurrent threads are supported.\n";

	typedef float data_type;
	typedef NeuralNetwork<data_type> float_network;
	typedef DataSet<data_type> float_dataset;
	typedef MultithreadTrainer<float_network, float_dataset> multi_trainer;
	// typedef Trainer<float_network, float_dataset> float_trainer;

	float_network ann({784, 800, 10}, ActFn::Tanh, ActFn::Softmax);
	float_dataset ds_train(true, "data/img_train", "data/label_train", 60000);
	float_dataset ds_test(true, "data/img_test", "data/label_test", 10000);

	// float_dataset ds_train("data/xor8.txt", 256, 8, 1);	
	// float_trainer trainer(ann, ds_train, 0.01f, 0.9f, ErrFn::CrossEntropy);
	// trainer.set_test_data(&ds_test);
	// trainer.train_batch(100, 100);
	// trainer.train_rprop(100);
	// trainer.train_rprop2(100);
	// trainer.train_adagrad(100);
	// trainer.train_quickprop(100);
	multi_trainer mt(ann, ds_train, 4, 0.01f, 0.9f, ErrFn::CrossEntropy);
	mt.set_test_data(ds_test);
	mt.train(200, 50);

	// vector<data_type> input = {0, 0, 0, 0, 0, 0, 0, 0.0};
	// st.feedforward(input);
	// cout<<st.output_act[0];

	return 0;
}
