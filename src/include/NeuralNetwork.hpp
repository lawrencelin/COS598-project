#ifndef NN_NEURALNETWORK_H
#define NN_NEURALNETWORK_H

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
#include <thread>

namespace NN
{
	using namespace std;

	template <class T>
	void print_V2D(ostream& os, const vector<vector<T>> & v2d){
		for (auto & v1d : v2d){
			for (auto & t: v1d) os<<t<<" ";
			cerr<<"\n";
		}
	}
	template <class T>
	void print_V3D(ostream& os, const vector<vector<vector<T>>> & v3d){
		for (auto & v2d : v3d){
			for (auto & v1d : v2d){
				for (auto & t: v1d) os<<t<<" ";
				os<<"\n";
			}
			os<<"==================\n";
		}
	}
	enum class ActFn
	{
		Sigmoid = 1,
		Tanh,
		Linear
	};
	template <class Ty>
	class NeuralLayer
	{
	private:
		size_t in_sz;
		size_t sz;
		vector<vector<Ty>> w; // weight
		vector<Ty> b; // bias
		ActFn act;
		typedef typename vector<vector<Ty>>::iterator neuron_iter;
		typedef typename vector<Ty>::iterator bias_iter;
		typedef typename vector<vector<Ty>>::const_iterator const_neuron_iter;
		class column_iterator : public iterator<input_iterator_tag, Ty>
		{
		private:
			size_t column;
			const_neuron_iter it;
		public:
			column_iterator(const_neuron_iter _it, size_t idx) : column(idx), it(_it) {}
			column_iterator(const column_iterator& cit) : column(cit.column), it(cit.it) {}
			column_iterator& operator++() {++it;return *this;}
			column_iterator operator++(int) {column_iterator tmp(*this); operator++(); return tmp;}
			bool operator==(const column_iterator& rhs) {return column==rhs.column && it == rhs.it;}
			bool operator!=(const column_iterator& rhs) {return !(*this == rhs);}
			const Ty& operator*() {return it->at(column); }
		};

	public:
		void print(){
			switch (act){
				case ActFn::Sigmoid: cout<<"Sigmoid"; break;
				case ActFn::Tanh: cout<<"Tanh"; break;
				case ActFn::Linear: cout<<"Linear"; break;
			}
			cout<<"Layer "<<in_sz<<" input, "<<sz<<" output.\nWeight:\n";
			for (auto &w_j : w){
				for (auto w_ij: w_j) cout<<w_ij<<" ";
				cout<<"\n";
			}
			cout<<"Bias:\n";
			for (auto b_j : b) cout<<b_j<<" ";
			cout<<"\n";
		}
		typedef Ty data_type;
		NeuralLayer(size_t in, size_t out, ActFn _act = ActFn::Sigmoid)
		: in_sz(in), sz(out), w(out, vector<Ty>(in)), b(out), act(_act)
		{
			Ty r = sqrt((Ty)6.0/(in + out));
			
			mt19937 gen((random_device()()));
			uniform_real_distribution<Ty> dist(-r, r);

			for (auto &w_i : w) for (Ty &w_ij : w_i) w_ij = dist(gen);

			for (Ty &b_i : b) b_i = dist(gen);
		}
		bool operator==(const NeuralLayer& rhs) {return w==rhs.w && b==rhs.b && act == rhs.act;}
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

		Ty act_fn(Ty x) {
			switch (act){
				case ActFn::Sigmoid: return (Ty)1.0/((Ty)1.0 + exp(-x));
				case ActFn::Tanh: return tanh(x);
				case ActFn::Linear: return x;
			}
		}
		Ty act_fn_derived(Ty fx) {
			switch (act){
				case ActFn::Sigmoid: return (Ty)fx * ((Ty)1.0 - fx);
				case ActFn::Tanh: return (Ty)1.0 - fx * fx;
				case ActFn::Linear: return (Ty)1.0;
			}
	}

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
		
		// For iterating the neural network backwards
		class reverse_adapter{
		private:
			vector<layer_type>& layers;
			typedef typename vector<layer_type>::const_reverse_iterator rev_iter;
		public:
			reverse_adapter(vector<layer_type>& _layers): layers(_layers) {}
			rev_iter begin() const {return layers.crbegin();}
			rev_iter end() const { return layers.crend();}
		};

	public:
		typedef Ty data_type;
		typedef typename layer_type::pointer layer_ptr;
		void print() {for (auto &layer: layers) layer->print();}
		NeuralNetwork(const initializer_list<size_t> l)
		: layers((
			assert("NeuralNetwork must be initilized with at least one non-input layer!" && l.size() > 1)
			,l.size() - 1))
		{
			size_t in, i = 0;
			for (size_t out : l) {
				if (i) layers[i-1] = make_unique<NeuralLayer<Ty>>(in, out, (i + 1 == l.size()) ? ActFn::Linear : ActFn::Tanh);
				in = out;
				++i;
			}
		}
		NeuralNetwork(const NeuralNetwork& rhs) 
		: layers(rhs.size()) {
			for (size_t i = 0; i < layers.size(); ++i) layers[i] = make_unique<NeuralLayer<Ty>>(*rhs.layers[i]);
		}
		bool operator==(const NeuralNetwork & rhs) {
			bool eq = true;
			for (size_t i = 0; i < layers.size(); ++i)
				eq = eq && *(layers[i]) == *(rhs.layers[i]);
			return eq;
		}
		// virtual ~NeuralNetwork() {}
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
		DataSet(const char *f, size_t n, size_t _in_sz, size_t _out_sz)  // if n * (in_sz + out_sz) > #of data in file, this function crashes.
		:	in_sz(_in_sz), out_sz(_out_sz), 
			data(n, sample(piecewise_construct, forward_as_tuple(in_sz), forward_as_tuple(out_sz)))
		{
			ifstream fs(f);
			assert("Cannot open file" && fs.is_open());
			istream_iterator<Ty> file_reader(fs);
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
	class Trainer
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

		bool multi_thread;
		
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
			return inner_product(target.cbegin(), target.cend(), output_act.cbegin(), (data_type)0.0, plus<data_type>(), 
				[](data_type t, data_type o){ return (t-o)*(t-o);}) / (data_type)target.size();
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
					else slope[l][j] = inner_product(slope[l+1].begin(), slope[l+1].end(), next_layer->w_col_iter(j), (data_type)0.0) * act_derived;
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
		
		void apply_dw(size_t n_samples)
		{
			data_type c = learning_rate / n_samples;

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

		int sign(data_type x) {
			if (x < 0) return -1;
			if (x < 0.00000000000000001) return 0; // default zero tolerance
			return 1;
		}

		void apply_dw_rprop( 
			V3D_ptr& delta_weight, V3D_ptr& prev_gradient_w, 
			V2D_ptr& delta_bias, V2D_ptr& prev_gradient_b, 
			data_type delta_max) /* rprop; used only in batch learning */
		{
			//cout<<"df: "<<(*delta_weight)[1][1][1]<<"\n";

			// values from original paper
			data_type delta_min = 1e-10;
			data_type eta_plus = 1.2, eta_minus = 0.5; 
			size_t l = 0;
			for (auto &layer: ann){
				size_t j = 0;
				auto b_j = layer->bias_begin();
				for (auto &w_j: *layer){
					size_t i = 0;
					for (auto & w_ij: w_j){
						auto& prev_gradient_w_ = (*prev_gradient_w)[l][j][i];
						data_type cur_gradient_w = (*partial_dw)[l][j][i]; // the newly calculated gradient dE/dw
						int change = sign(prev_gradient_w_ * cur_gradient_w);
						if (change > 0) { // same sign
							//cout<<"same sign\n";
							(*delta_weight)[l][j][i] = min((*delta_weight)[l][j][i] * eta_plus, delta_max); // update delta
							data_type dw = sign((*partial_dw)[l][j][i]) * (*delta_weight)[l][j][i];
							w_ij -= dw; // update weight
							prev_gradient_w_ = cur_gradient_w;
						} else if (change < 0) { // change sign: last delta was too big
							//w_ij -= (*delta_weight)[l][j][i]; // revert weight
							(*delta_weight)[l][j][i] = max((*delta_weight)[l][j][i] * eta_minus, delta_min); // update delta
							prev_gradient_w_ = 0; // set previous gradient to zero so that there will be no adjustment to delta next round
						} else { // zero: no change in delta
							data_type dw = sign((*partial_dw)[l][j][i]) * (*delta_weight)[l][j][i];
							w_ij -= dw; // update weight
							prev_gradient_w_ = cur_gradient_w;
						}
						++i;
					}

					auto& prev_gradient_b_ = (*prev_gradient_b)[l][j];
					data_type gradient = (*partial_db)[l][j];
					if (prev_gradient_b_ * gradient > 0) {
						(*delta_bias)[l][j] = min((*delta_bias)[l][j] * eta_plus, delta_max);
						data_type db = (-1) * sign((*partial_db)[l][j]) * (*delta_bias)[l][j];
						*b_j += db;
						prev_gradient_b_ = (*partial_db)[l][j];
					} else if (prev_gradient_b_ * gradient < 0) {
						(*delta_bias)[l][j] = max((*delta_bias)[l][j] * eta_minus, delta_min);
						prev_gradient_b_ = 0;
					} else {
						data_type db = (-1) * sign((*partial_db)[l][j]) * (*delta_bias)[l][j];
						*b_j += db;
						prev_gradient_b_ = (*partial_db)[l][j];
					}
					++j, ++b_j;
				}
				++l;
			}
		}

		void print_progress(size_t i, size_t epochs) {
			size_t one_percent = epochs / 100;
			double percentage = 100.0 * i / epochs;
			if (!one_percent || i % one_percent == 0)
				cout<<percentage<<"% epoch "<<i<<": error "<<compute_batch_error()<<"\n";
		}
		data_type train_epilogue() {
			sum_err = 0.0;
			for (auto &data : data_range){
				feedforward(data.first);
				cout<<"test---";
				for( auto v :data.first) cout<<v<<" ";
				cout<<"answer:";
				for( auto v :output_act) cout<<v<<" ";
					cout<<" should be:";
				for( auto v :data.second) cout<<v<<" ";
					cout<<"\n";
				sum_err += compute_error(data.second);
			}
			ann.print();
			return compute_batch_error();
		}
		void fill_zero_2d(V2D &v2d) {for (auto &v1d:v2d) fill(v1d.begin(), v1d.end(), 0.0); }
		void fill_zero_3d(V3D &v3d) {for (auto &v2d:v3d) fill_zero_2d(v2d); }

		void fill_init_value_2d(V2D &v2d, data_type initial_value) {for (auto &v1d:v2d) fill(v1d.begin(), v1d.end(), initial_value); }
		void fill_init_value_3d(V3D &v3d, data_type initial_value) {for (auto &v2d:v3d) fill_init_value_2d(v2d, initial_value); }

		void init_vectors(const vector<V2D*>& init_list_2d, const vector<V3D*>& init_list_3d) {
			size_t l = 0;
			for (auto &layer : ann){
				for (V2D *v: init_list_2d)
					v->at(l).resize(layer->size());
				for (V3D *v: init_list_3d)
					v->at(l).resize(layer->size(), V1D(layer->input_size()));
				++l;
			}
		}
	public:
		Trainer(ANN &_ann, const DataRange &_data_range, data_type alpha = (data_type)0.01, data_type beta = (data_type)0.9)
		:	ann(_ann), act(ann.size()), output_act(act[ann.size()-1]), slope(ann.size()), 
			prev_dw(make_unique<V3D>(ann.size())), partial_dw(make_unique<V3D>(ann.size())),
			prev_db(make_unique<V2D>(ann.size())), partial_db(make_unique<V2D>(ann.size())),
			data_range((assert("Dataset must not be empty" && _data_range.size()), _data_range)),
			learning_rate(alpha), momentum(beta), multi_thread(false)
		{
			init_vectors(
				{&act, &slope, prev_db.get(), partial_db.get()},
				{prev_dw.get(), partial_dw.get()});
		}

		// construct multi-thread trainer;
		Trainer(bool placeholer, ANN &_ann, const DataRange &_data_range)
		:	ann(_ann), act(ann.size()), output_act(act[ann.size()-1]), slope(ann.size()), 
			partial_dw(make_unique<V3D>(ann.size())), partial_db(make_unique<V2D>(ann.size())),
			data_range((assert("Dataset must not be empty" && _data_range.size()), _data_range)), multi_thread(true)
		{
			placeholer = true;
			init_vectors(
				{&act, &slope, partial_db.get()},
				{partial_dw.get()});
		}

		Trainer(const Trainer& rhs) = delete;
		Trainer(Trainer& rhs) = delete;

		data_type train_stochastic(size_t epochs)
		{
			assert("multi-thread trainer" && !multi_thread);
			fill_zero_3d(*prev_dw);
			fill_zero_2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i) {
				sum_err = 0;
				for (auto &data : data_range){
					fill_zero_3d(*partial_dw);
					fill_zero_2d(*partial_db);
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					apply_dw(1);
				}
				print_progress(i, epochs);
			}
			return train_epilogue();
		}

		data_type train_minibatch(size_t epochs, size_t batch_sz)
		{
			assert("multi-thread trainer" && !multi_thread);
			fill_zero_3d(*prev_dw);
			fill_zero_2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i) {
				sum_err = 0;
				auto data_iter = data_range.begin();
				auto data_end = data_range.end();
				size_t cnt = 0;
				while (data_iter != data_end){
					fill_zero_3d(*partial_dw);
					fill_zero_2d(*partial_db);
					feedforward(data_iter->first);
					backpropagate(data_iter->first, data_iter->second);
					sum_err += compute_error(data_iter->second);					
					++data_iter, ++cnt;
					if (cnt == batch_sz || data_iter == data_end){
						apply_dw(cnt);
						cnt = 0;
					}
				}
				print_progress(i, epochs);
			}
			return train_epilogue();
		}

		data_type train_batch(size_t epochs) // return final error
		{
			assert("multi-thread trainer" && !multi_thread);
			fill_zero_3d(*prev_dw);
			fill_zero_2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_zero_3d(*partial_dw);
				fill_zero_2d(*partial_db);
				int k = 0;
				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					k++;
				}
				print_progress(i, epochs);
				apply_dw(data_range.size());
			}
			return train_epilogue();
		}

		data_type train_batch_rprop(size_t epochs, data_type delta_initial = 0.1, data_type delta_max = 50.0) // rprop algorithmn based on original paper
		{
			assert("multi-thread trainer" && !multi_thread);

			fill_zero_3d(*prev_dw);
			fill_zero_2d(*prev_db);

			// rprop specific
			V3D_ptr prev_gradient_w = make_unique<V3D>(ann.size()); // dE/dw_ij(t-1)
			V2D_ptr prev_gradient_b = make_unique<V2D>(ann.size()); // dE/db_i(t-1)
			V3D_ptr delta_weight = make_unique<V3D>(ann.size()); // delta_ij for weight
			V2D_ptr delta_bias = make_unique<V2D>(ann.size()); // delta_ij for bias
			init_vectors(
				{prev_gradient_b.get(), delta_bias.get()},
				{prev_gradient_w.get(), delta_weight.get()});

			fill_zero_3d(*prev_gradient_w);
			fill_zero_2d(*prev_gradient_b); 
			fill_init_value_3d(*delta_weight, delta_initial);
			fill_init_value_2d(*delta_bias, delta_initial);

			//cout<<(*delta_weight)[0][0][0];
			//return 0.1;

			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_zero_3d(*partial_dw);
				fill_zero_2d(*partial_db);
				int k = 0;
				//cout<<"ha: " <<(*delta_weight)[1][1][1]<<"\n";
				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					k++;
				}
				print_progress(i, epochs);
				apply_dw_rprop(delta_weight, prev_gradient_w, delta_bias, prev_gradient_b, delta_max);
				//if (i == 2) return 0;
			}
			return train_epilogue();
		}

		// return sum of errors
		data_type train_multi_thread()
		{
			assert("multi-thread trainer" && multi_thread);
			fill_zero_3d(*partial_dw);
			fill_zero_2d(*partial_db);
			sum_err = 0;
			for (auto &data : data_range){
				feedforward(data.first);
				backpropagate(data.first, data.second);
				sum_err += compute_error(data.second);
			}
			return sum_err;
		}
		data_type train_epilogue_multi_thread() {
			sum_err = 0.0;
			for (auto &data : data_range){
				feedforward(data.first);
				cout<<"test---";
				for( auto v :data.first) cout<<v<<" ";
				cout<<"answer:";
				for( auto v :output_act) cout<<v<<" ";
					cout<<" should be:";
				for( auto v :data.second) cout<<v<<" ";
					cout<<"\n";
				sum_err += compute_error(data.second);
			}
			return sum_err;
		}
		typedef typename V2D::const_iterator V2D_iter;
		typedef typename V3D::const_iterator V3D_iter;

		V3D_iter dw_begin() const { return partial_dw->cbegin(); }
		V3D_iter dw_end() const { return partial_dw->cend(); }
		V2D_iter db_begin() const { return partial_db->cbegin(); }
		V2D_iter db_end() const { return partial_db->cend(); }
	};

	template<class T>
	class sub_range {
	private:
		typedef typename T::sample_iter iter;
		const T &t;
		size_t start, sz;
	public:
		sub_range(const T &_t, size_t _start, size_t _sz)
		: t(_t), start(_start), sz(_sz) {}
		sub_range(const sub_range&) = delete;
		iter begin() const { return t.begin() + (long)start; }
		iter end() const { return t.begin() + (long)start + (long)sz;}
		size_t size() const { return sz; }
	};

	enum class JobCode{
		NOP = 0,
		Compute = 1,
		Exit = 2
	};
	enum class ReplyCode{
		NOP = 0,
		Completed = 1
	};

	inline ostream& operator<<(ostream &os, const enum JobCode j) {
		string str;
		switch (j){
			case JobCode::NOP: str = "NOP"; break;
			case JobCode::Compute: str = "Compute"; break;
			case JobCode::Exit: str = "Exit"; break;
		}
		return os<<"JobCode::"+str;
	}

	inline ostream& operator<<(ostream &os, const enum ReplyCode r) {
		string str;
		switch (r){
			case ReplyCode::NOP: str = "NOP"; break;
			case ReplyCode::Completed: str = "Completed"; break;
		}
		return os<<"ReplyCode::"+str;
	}

	template<class First, class Second>
	inline ostream& operator<<(ostream &os, const pair<First, Second> &p) {
		return os<<"("<<p.first<<", "<<p.second<<")";
	}

	template<class T>
	class sync_channel{
	private:
		condition_variable cv;
		mutex cv_m;
		string name;
		vector<T> v;
	public:
		sync_channel(const string _name, const size_t _n): name(_name), v(_n) {
			assert("Must not be empty!" && _n);
		}
		void set(size_t k, T t = T()) {
			unique_lock<mutex> l(cv_m);
			v[k] = t;
			cv.notify_all();
		}
		void set_all(T t = T()) {
			unique_lock<mutex> l(cv_m);
			fill(v.begin(), v.end(), t);
			cv.notify_all();
		}
		T get(size_t k) {
			unique_lock<mutex> l(cv_m);
			cv.wait(l, [this, k](){return this->v[k] != T();});
			T t = v[k];
			v[k] = (T)0;
			return t;
		}
		vector<T> get_all() {
			unique_lock<mutex> l(cv_m);
			cv.wait(l, [this](){return count(this->v.begin(), this->v.end(), T()) == 0;});
			vector<T> ret = v;
			fill(v.begin(), v.end(), T());
			return ret;
		}
	};

	template <class ANN, class DataRange>
	class MultithreadTrainer 
	{
	private:
		typedef typename ANN::data_type data_type;
		typedef vector<data_type> V1D;
		typedef vector<V1D> V2D;
		typedef vector<V2D> V3D;
		typedef sub_range<DataRange> range;
		typedef Trainer<ANN, range> single_trainer;
		
		ANN &ann;
		const DataRange& full_range;

		V3D prev_dw;
		V2D prev_db;

		data_type learning_rate;
		data_type momentum;

		vector<unique_ptr<range>> ranges; // need heap storage to prevent worker threads accessing corrupted (out-dated) stack frames
		vector<unique_ptr<single_trainer>> trainers; // same heap storage
		vector<thread> threads;

		typedef pair<ReplyCode, data_type> reply;
		sync_channel<JobCode> job_chan;
		sync_channel<reply> reply_chan;

		void worker_thread_loop(single_trainer *trainer, size_t id) {
			while (1){
				switch(job_chan.get(id)){
					case JobCode::NOP: assert("received NOP" && false);
				case JobCode::Compute:
						reply_chan.set(id, reply(ReplyCode::Completed, trainer->train_multi_thread()));
						break;
					case JobCode::Exit:
						return;
				}
			}
		}

		void apply_dw() 
		{
			data_type c = learning_rate / full_range.size();
			size_t l = 0;
			typedef typename single_trainer::V3D_iter V3D_iter;
			typedef typename single_trainer::V2D_iter V2D_iter;
			vector<V3D_iter> dw_iters;
			vector<V2D_iter> db_iters;
			transform(trainers.begin(), trainers.end(), back_inserter(dw_iters), [](const auto& t){return t->dw_begin();});
			transform(trainers.begin(), trainers.end(), back_inserter(db_iters), [](const auto& t){return t->db_begin();});
			for (auto &layer: ann){
				size_t j = 0;
				auto bj_iter = layer->bias_begin();
				vector<V2D_iter> dwj_iters;
				transform(dw_iters.begin(), dw_iters.end(), back_inserter(dwj_iters), [j](const auto& it){ return it->cbegin(); });
				for (auto &wj: *layer){
					size_t i = 0;
					for (auto & wij: wj){
						data_type sum_dw = 0;
						for (auto & it: dwj_iters) sum_dw += it->at(i);
						wij -= (prev_dw[l][j][i] = c * sum_dw + momentum * prev_dw[l][j][i]);
						++i;
					}
					data_type sum_db = 0;
					for (auto & it: db_iters) sum_db += it->at(j);
					*bj_iter -= (prev_db[l][j] = c * sum_db + momentum * prev_db[l][j]);
					for (auto & it: dwj_iters) ++it;
					++j, ++bj_iter;
				}
				for (auto & it: dw_iters) ++it;
				for (auto & it: db_iters) ++it;
				++l;
			}
		}
		void print_progress(size_t i, size_t epochs, data_type avg_err) {
			size_t one_percent = epochs / 100;
			double percentage = 100.0 * i / epochs;
			if (!one_percent || i % one_percent == 0)
				cout<<percentage<<"% epoch "<<i<<": error "<<avg_err<<"\n";
		}
	public:
		MultithreadTrainer(ANN &_ann, const DataRange &data_range, size_t n_t = 4,
			data_type alpha = (data_type)0.01, data_type beta = (data_type)0.9)
		:	ann(_ann), full_range(data_range), prev_dw(ann.size()), prev_db(ann.size()),
			learning_rate(alpha), momentum(beta), ranges(n_t), trainers(n_t), threads(n_t),
			job_chan("{main->worker}", n_t), reply_chan("{worker->main}", n_t)
		{
			size_t n_hw_t = thread::hardware_concurrency();
			if (n_hw_t < n_t) cerr<<"WARNING: requesting more threads("<<n_t<<") than the hardware supports("<<n_hw_t<<").\n";
			size_t sub_range_sz = data_range.size() / n_t;
			assert("more threads than data samples" && sub_range_sz);
			
			size_t l = 0;
			for (auto &layer : ann){
				prev_db[l].resize(layer->size());
				prev_dw[l].resize(layer->size(), V1D(layer->input_size()));
				++l;
			}
			for (size_t i = 0; i < n_t; ++i) {
				ranges[i] = make_unique<range>(full_range, i * sub_range_sz, min(sub_range_sz, full_range.size() - i * sub_range_sz));
				trainers[i] = make_unique<single_trainer>(true, ann, *(ranges[i].get()));
				threads[i] = thread(&MultithreadTrainer::worker_thread_loop, this, trainers[i].get(), i);
				// pointers in ranges / thrainers will outlive threads
				// because they are deleted after the destructor ~MultithreadTrainer which joins the threads
			}
		}
		data_type train(size_t epochs) 
		{
			data_type sum_err = 0;
			for (size_t i = 0; i < epochs; ++i) {
				job_chan.set_all(JobCode::Compute);
				vector<reply> r = reply_chan.get_all();
				sum_err = 0;
				for (auto& rep : r){
					assert("bad job" && rep.first == ReplyCode::Completed);
					sum_err += rep.second;
				}
				apply_dw();
				print_progress(i, epochs, sqrt(sum_err / full_range.size()));
			}
			sum_err = 0;
			for (auto &t : trainers) sum_err += t->train_epilogue_multi_thread(); 
			ann.print();
			return sqrt(sum_err / full_range.size());
		}
		~MultithreadTrainer() 
		{
			job_chan.set_all(JobCode::Exit);
			for (thread &worker : threads) worker.join();
		}
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

#endif /*NN_NEURALNETWORK_H*/
