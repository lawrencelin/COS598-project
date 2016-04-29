#ifndef __NEURALNETWORK_H
#define __NEURALNETWORK_H

#include <initializer_list>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <fstream>
#include <iterator>
#include <iostream>
#include <utility>
#include <cassert>
#include <tuple>

namespace NN
{
	using namespace std;
	
	template <class Ty>
	class NeuralLayer
	{
	private:
		size_t sz;
		// bool bias;
		vector<vector<Ty>> w; // weight

		vector<Ty> b; // bias
	public:
		typedef Ty data_type;
		// bool hasBiase() {return bias; }
		// int size() {return sz + (bias ? 1: 0); }
		NeuralLayer(size_t in, size_t out)
		: sz(out), w(out, vector<Ty>(in)), b(out)
		{
			Ty r = sqrt(6.0/(in + out));
			
			mt19937 gen((random_device()()));
			uniform_real_distribution<Ty> dist(-r, r);

			for (auto &w_i : w)
				for (Ty &w_ij : w_i)
					w_ij = dist(gen);

			for (Ty &b_i : b)
				b_i = dist(gen);
		}
		~NeuralLayer() {};
		size_t size() {return sz;}



	};
	
	template <class Ty>
	class NeuralNetwork
	{
	private:
		// Not counting input layers: input x hidden x output = 2 layers.
		typedef unique_ptr<NeuralLayer<Ty> > layer_type;
		vector<layer_type> layers;
	public:
		typedef Ty data_type;
		NeuralNetwork(initializer_list<size_t> l)
		: layers((
			assert("NeuralNetwork must be initilized with at least one non-input layer!" && l.size() > 1)
			,l.size() - 1))
		{
			bool _1st = true;
			size_t in, i = 0;
			for (size_t out : l) {
				if (_1st) {
					in = out;
					_1st = false;
					continue;
				}
				layers[i++] = std::make_unique<NeuralLayer<Ty>>(in, out);
				in = out;
			}
		}
		~NeuralNetwork() {};
		// int size() {return layers.size();}
	};

	template <class Ty>
	class DataSet
	{
	private:
		typedef pair<vector<Ty>, vector<Ty>> DataPt;
		int in_sz;
		int out_sz;
		vector<DataPt> data;
	public:
		typedef typename vector<DataPt>::const_iterator DataPtIter;
		DataSet(const char *f, int n, int in_sz, int out_sz)  // if n * (in_sz + out_sz) > #of data in file, this function crashes.
		:	in_sz(in_sz), out_sz(out_sz), 
			data(n, DataPt(piecewise_construct, forward_as_tuple(in_sz), forward_as_tuple(out_sz)))
		{
			ifstream fs(f);
			assert("Cannot open file" && fs.is_open());
			istream_iterator<Ty> read_file(fs);
			// istreambuf_iterator<Ty> end_file;
			for (DataPt &dp : data){
				for (Ty &ref : dp.first)
					ref = *(read_file++);
				for (Ty &ref : dp.second)
					ref = *(read_file++);
			}
		}
		DataPtIter begin() const { return data.cbegin(); }
		DataPtIter end() const { return data.cend(); }
		size_t size() {return data.size(); }
	};

	template <class ANN, class DataRange>
	class BatchTrainer
	{
	private:
		ANN &ann;
		typedef typename ANN::data_type data_type;
		const DataRange &data_range;
		size_t n_data;
		data_type sum_sq_err;


		template <class Input>
		void feedforward(const Input &input)
		{

		}

		template <class Output>
		data_type compute_MSE(const Output &target)
		{
			return 0.0;
		}

		template <class Output>
		void backpropagate(const Output &target)
		{

		}
		
		data_type compute_batch_RMSE()
		{
			return sqrt(sum_sq_err / n_data);
		}
		
		void apply_deltaw()
		{

		}
	public:
		BatchTrainer(ANN &ann, const DataRange &data_range)
		: ann(ann), data_range(data_range)
		{
			cout<<"Trainer initilized.\n";
		}
		data_type train(int epochs) // return final RSME
		{
			for (int i = 0; i < epochs; ++i){
				sum_sq_err = 0;
				for (auto &data : data_range){
					this->feedforward(data.first);
					this->backpropagate(data.second);
				}
				cout<<"Epoch "<<(i+1)<<": RMSE "<<compute_batch_RMSE()<<"\n";
				this->apply_deltaw();
			}

			sum_sq_err = 0.0;
			for (auto &data : data_range){
				this->feedforward(data.first);
				sum_sq_err += this->compute_MSE(data.second);
			}
			return compute_batch_RMSE();
		}

		~BatchTrainer() {}
		
	};

}


// void train(){
// 	// d(E)/d(w)  = d(E)/d(phi) * d(phi)/d(net) * d(net)/d(w)

// 	// mini-batch to threads

// 	feedforward(input); // compute activations (phi) and (phi)'
// 	// mini-batch to threads
// 	backpropagate(output); // compute d(E)/d(phi) = sum(d(E)/d(phi) * w) and update delta(w)

// 	// back to main thread
// 	updateweights(); // w += sum(delta(w)) * learning_rate / n OR rprop / quickprop


// }

#endif /*__NEURALNETWORK_H*/