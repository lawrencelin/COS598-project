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
#include <numeric>

namespace NN
{
	using namespace std;
	
	template <class Ty>
	class NeuralLayer
	{
	private:
		size_t in_sz;
		size_t sz;
		vector<vector<Ty>> w; // weight
		vector<Ty> b; // bias
		typedef typename vector<vector<Ty>>::iterator neuron_iter;
	public:
		typedef Ty data_type;
		NeuralLayer(size_t in, size_t out)
		: in_sz(in), sz(out), w(out, vector<Ty>(in)), b(out)
		{
			Ty r = sqrt(6.0/(in + out));
			
			mt19937 gen((random_device()()));
			uniform_real_distribution<Ty> dist(-r, r);

			for (auto &w_i : w) for (Ty &w_ij : w_i) w_ij = dist(gen);

			for (Ty &b_i : b) b_i = dist(gen);
		}
		virtual ~NeuralLayer() {}
		size_t size() const {return sz;}
		size_t input_size() const  {return in_sz;}

		neuron_iter begin() const {return w.begin(); }
		neuron_iter end() const {return w.end(); }

		template <class InputIter, class OutputIter>
		void feedforward(InputIter in, OutputIter out)
		{
			auto b_iter = b.begin();
			for (auto &w_i : w)
				*(out++) = act_fn(inner_product(w_i.begin(), w_i.end(), in, (data_type)0.0) + *(b_iter++));
		}

		Ty act_fn(Ty x) { return (Ty)1.0/(1.0 + exp(-x)); }
		// Ty act_fn_derived(Ty fx) { return (Ty)fx * (1.0 - fx); }






	};
	
	template <class Ty>
	class NeuralNetwork
	{
	private:
		// Not counting input layers: input x hidden x output = 2 layers.
		typedef unique_ptr<NeuralLayer<Ty> > layer_type; // using pointers to allow classes derived from NeuralLayer
		typedef typename vector<layer_type>::const_iterator layer_iter;
		vector<layer_type> layers;
		

		class reverse_adapter{
		private:
			vector<layer_type>& layers;
			typedef typename vector<layer_type>::const_reverse_iterator rev_iter;
		public:
			reverse_adapter(vector<layer_type>& layers): layers(layers) {}
			rev_iter begin() const {return layers.crbegin();}
			rev_iter end() const { return layers.crend();}
		};
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
		virtual ~NeuralNetwork() {}
		layer_iter begin() const { return layers.cbegin(); }
		layer_iter end() const { return layers.cend(); }
		size_t size() const {return layers.size();}
		reverse_adapter rev() {return reverse_adapter(layers);}
	};

	template <class Ty>
	class DataSet
	{
	private:
		typedef pair<vector<Ty>, vector<Ty>> sample;
		typedef typename vector<sample>::const_iterator sample_iter;
		size_t in_sz;
		size_t out_sz;
		vector<sample> data;
	public:
		DataSet(const char *f, size_t n, size_t in_sz, size_t out_sz)  // if n * (in_sz + out_sz) > #of data in file, this function crashes.
		:	in_sz(in_sz), out_sz(out_sz), 
			data(n, sample(piecewise_construct, forward_as_tuple(in_sz), forward_as_tuple(out_sz)))
		{
			ifstream fs(f);
			assert("Cannot open file" && fs.is_open());
			istream_iterator<Ty> file_reader(fs);
			// istreambuf_iterator<Ty> end_file;
			for (sample &dp : data){
				for (Ty &ref : dp.first)
					ref = *(file_reader++);
				for (Ty &ref : dp.second)
					ref = *(file_reader++);
			}
		}
		sample_iter begin() const { return data.cbegin(); }
		sample_iter end() const { return data.cend(); }
		size_t size() const {return data.size(); }
	};

	template <class ANN, class DataRange>
	class BatchTrainer
	{
	private:
		ANN &ann;
		typedef typename ANN::data_type data_type;

		vector<vector<data_type>> act; // act[i][j] = activation of layer_i, neural_j
		vector<data_type> &output_act;
		vector<vector<data_type>> slope; // slope[i][j] = d(Err)/d(input_j) of layer_i, neural_j
		vector<vector<vector<data_type>>> partial_dw; // partial_dw[i][j][k] = d(k-th input weight) of layer_i, neural_j
		const DataRange &data_range;
		data_type sum_sq_err;


		template <class Input>
		void feedforward(const Input &input)
		{
			bool _1st = true;
			auto act_iter = act.begin();
			for (auto &layer: ann) {
				// for each layer
				if (_1st) layer->feedforward(input.begin(), act_iter->begin());
				else      layer->feedforward((act_iter - 1)->begin(), act_iter->begin());
				_1st = false;
				++act_iter;
			}
		}

		template <class Output>
		data_type compute_MSE(const Output &target)
		{
			return inner_product(target.begin(), target.end(), output_act.begin(), 0.0, plus<data_type>(), 
				[](data_type t, data_type o){ return (t-o)*(t-o);}) / target.size();
		}

		template <class Output>
		void backpropagate(const Output &target)
		{
			for (auto &layer: ann.rev());
		}
		
		data_type compute_batch_RMSE()
		{
			return sqrt(sum_sq_err / data_range.size());
		}
		
		void apply_dw()
		{

		}
	public:
		BatchTrainer(ANN &ann, const DataRange &data_range)
		:	ann(ann), act(ann.size()), output_act(act[ann.size()-1]),
			slope(ann.size()), partial_dw(ann.size()), data_range(data_range)
		{
			
			// better with zip_iterator

			auto act_iter = act.begin();
			auto slope_iter = slope.begin();
			auto dw_iter = partial_dw.begin();
			for (auto &layer : ann){
				act_iter->resize(layer->size());
				slope_iter->resize(layer->size());
				dw_iter->resize(layer->size(), vector<data_type>(layer->input_size()));
				++act_iter, ++slope_iter, ++dw_iter;
			}
		}
		data_type train(size_t epochs) // return final RSME
		{
			for (size_t i = 0; i < epochs; ++i){
				sum_sq_err = 0;
				for (auto &data : data_range){
					this->feedforward(data.first);
					this->backpropagate(data.second);
				}
				cout<<"Epoch "<<(i+1)<<": RMSE "<<compute_batch_RMSE()<<"\n";
				this->apply_dw();
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