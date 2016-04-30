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
		typedef typename vector<Ty>::iterator bias_iter;
		typedef typename vector<vector<Ty>>::const_iterator const_neuron_iter;
		class column_iterator : public iterator<input_iterator_tag, Ty>
		{
		private:
			size_t column;
			const_neuron_iter it;
		public:
			column_iterator(const_neuron_iter it, size_t idx) : column(idx), it(it) {}
			column_iterator(const column_iterator& cit) : column(cit.column), it(cit.it) {}
			column_iterator& operator++() {++it;return *this;}
			column_iterator operator++(int) {column_iterator tmp(*this); operator++(); return tmp;}
			bool operator==(const column_iterator& rhs) {return column==rhs.column && it == rhs.it;}
			bool operator!=(const column_iterator& rhs) {return !(*this == rhs);}
			const Ty& operator*() {return it->at(column); }
		};
	public:
		typedef Ty data_type;
		NeuralLayer(size_t in, size_t out)
		: in_sz(in), sz(out), w(out, vector<Ty>(in)), b(out)
		{
			Ty r = sqrt(6.0/(in + out));
			// Ty r = 5;
			
			mt19937 gen((random_device()()));
			uniform_real_distribution<Ty> dist(-r, r);

			for (auto &w_i : w) for (Ty &w_ij : w_i) w_ij = dist(gen);

			for (Ty &b_i : b) b_i = dist(gen);
		}
		virtual ~NeuralLayer() {}
		size_t size() const {return sz;}
		size_t input_size() const  {return in_sz;}

		neuron_iter begin() {return w.begin(); }
		neuron_iter end() {return w.end(); }

		bias_iter bias_begin() {return b.begin(); }
		bias_iter bias_end() {return b.end(); }

		template <class InputIter, class OutputIter>
		void feedforward(InputIter in, OutputIter out)
		{
			auto b_iter = b.cbegin();
			for (auto &w_i : w)
				*(out++) = act_fn(inner_product(w_i.cbegin(), w_i.cend(), in, (data_type)0.0) + *(b_iter++));
		}

		Ty act_fn(Ty x) { return (Ty)1.0/(1.0 + exp(-x)); }
		Ty act_fn_derived(Ty fx) { return (Ty)fx * (1.0 - fx); }

		column_iterator w_col_iter(size_t idx) { return column_iterator(w.cbegin(), idx);}
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
		typedef typename layer_type::pointer layer_ptr;
		NeuralNetwork(initializer_list<size_t> l)
		: layers((
			assert("NeuralNetwork must be initilized with at least one non-input layer!" && l.size() > 1)
			,l.size() - 1))
		{
			bool is_input_layer = true;
			size_t in, i = 0;
			for (size_t out : l) {
				if (is_input_layer) {
					in = out;
					is_input_layer = false;
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
		size_t in_sz;
		size_t out_sz;
	public:
		typedef pair<vector<Ty>, vector<Ty>> sample;
		typedef typename vector<sample>::const_iterator sample_iter;
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
	private:
		vector<sample> data;
	};

	template <class ANN, class DataRange>
	class BatchTrainer
	{
	private:
		ANN &ann;
		typedef typename ANN::data_type data_type;
		typedef vector<data_type> V1D;
		typedef vector<V1D> V2D;
		typedef vector<V2D> V3D;
		typedef unique_ptr<V2D> V2D_ptr;
		typedef unique_ptr<V3D> V3D_ptr;
		V2D act; // act[l][j] = activation of layer_l, neuron_j
		V1D &output_act; // reference to output layer activation
		V2D slope; // slope[l][j] = d(Err)/d(all_inputs_j) of layer_l, neuron_j
		
		V3D_ptr prev_dw;
		V3D_ptr partial_dw; // partial_dw[l][j][i] = delta (i-th input weight) of layer_l, neuron_j

		V2D_ptr prev_db;
		V2D_ptr partial_db; // partial_db[l][j] = delta (bias) of layer_l, neuron_j
		const DataRange &data_range;
		data_type sum_err;
		data_type learning_rate;
		data_type momentum;
		
		template <class Input>
		void feedforward(const Input &input)
		{
			size_t l = 0;
			for (auto &layer: ann) {
				if (l) layer->feedforward(act[l-1].cbegin(), act[l].begin());
				else layer->feedforward(input.cbegin(), act[l].begin());
				++l;
			}
		}

		template <class Output>
		data_type compute_error(const Output &target) // mean_sq_err
		{
			return inner_product(target.cbegin(), target.cend(), output_act.cbegin(), 0.0, plus<data_type>(), 
				[](data_type t, data_type o){ return (t-o)*(t-o);}) / target.size();
		}

		data_type compute_error_derived(const data_type o, const data_type t) { return (o-t); }

		template <class Input, class Output>
		void backpropagate(const Input &input, const Output &target)
		{
			typename ANN::layer_ptr next_layer = nullptr;
			size_t l = ann.size() - 1;
			for (auto &layer: ann.rev()){
				for (size_t j = 0; j < layer->size(); ++j) {
					data_type neuron_act = act[l][j];
					data_type act_derived = layer->act_fn_derived(neuron_act);
					if (l == ann.size() - 1) slope[l][j] = compute_error_derived(neuron_act, target[j]) * act_derived;
					else slope[l][j] = inner_product(slope[l+1].begin(), slope[l+1].end(), next_layer->w_col_iter(j), 0.0) * act_derived;
					for (size_t i = 0; i < layer->input_size(); ++i){
						if (l) (*partial_dw)[l][j][i] += slope[l][j] * act[l - 1][i];
						else (*partial_dw)[l][j][i] += slope[l][j] * input[i];
					}
					(*partial_db)[l][j] += slope[l][j];
				}
				next_layer = layer.get();
				--l;
			}
		}
		data_type compute_batch_error() { return sqrt(sum_err / data_range.size()); }
		
		void apply_dw()
		{
			data_type c = learning_rate / data_range.size();

			size_t l = 0;
			for (auto &layer: ann){
				size_t j = 0;
				auto b_j = layer->bias_begin();
				for (auto &w_j: *layer){
					size_t i = 0;
					for (auto & w_ij: w_j){
						auto& dw = (*partial_dw)[l][j][i];
						auto& dw_ = (*prev_dw)[l][j][i];
						dw = c * dw + momentum * dw_;
						w_ij -= dw;
						++i;
					}
					auto& db = (*partial_db)[l][j];
					auto& db_ = (*prev_db)[l][j];
					db = c * db + momentum * db_;
					*b_j -= db;
					++j, ++b_j;
				}
				++l;
			}
			
			prev_dw.swap(partial_dw);
			prev_db.swap(partial_db);
		}
	public:
		BatchTrainer(ANN &ann, const DataRange &data_range, data_type alpha = 0.01, data_type beta = 0.9)
		:	ann(ann), act(ann.size()), output_act(act[ann.size()-1]), slope(ann.size()), 
			prev_dw(make_unique<V3D>(ann.size())), partial_dw(make_unique<V3D>(ann.size())),
			prev_db(make_unique<V2D>(ann.size())), partial_db(make_unique<V2D>(ann.size())),
			data_range((assert("Dataset must not be empty" && data_range.size()), data_range)),
			learning_rate(alpha), momentum(beta)
		{
			vector<V2D*> init_list_2d = {&act, &slope, prev_db.get(), partial_db.get()};
			vector<V3D*> init_list_3d = {prev_dw.get(), partial_dw.get()};
			size_t l = 0;
			for (auto &layer : ann){
				for (V2D *v: init_list_2d)
					v->at(l).resize(layer->size());
				for (V3D *v: init_list_3d)
					v->at(l).resize(layer->size(), V1D(layer->input_size()));
				++l;
			}
		}
		data_type train(size_t epochs) // return final error
		{
			auto fill_zero_2d = [](V2D &v2d){for(auto &v1d:v2d)fill(v1d.begin(), v1d.end(), 0.0);};
			auto fill_zero_3d = [fill_zero_2d](V3D &v3d){for(auto &v2d:v3d)fill_zero_2d(v2d);};

			fill_zero_3d(*prev_dw);
			fill_zero_2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_zero_3d(*partial_dw);
				fill_zero_2d(*partial_db);
				
				for (auto &data : data_range){
					if (i == epochs - 1){
						cout<<"do something";
					}
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
				}
				cout<<"Epoch "<<(i+1)<<": error "<<compute_batch_error()<<"\n";
				apply_dw();
			}

			sum_err = 0.0;
			for (auto &data : data_range){
				feedforward(data.first);
				cout<<"test---";
				for( auto v :data.first) cout<<v;
				cout<<"answer:";
				for( auto v :output_act) cout<<v;
					cout<<" should be:";
				for( auto v :data.second) cout<<v;
					cout<<"\n";
				sum_err += compute_error(data.second);
			}
			return compute_batch_error();
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