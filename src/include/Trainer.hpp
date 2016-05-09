#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include <Utility.hpp>

namespace NN
{
	using namespace std;

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
		DataRange *test_range_ptr;
		data_type sum_err;
		data_type learning_rate;
		data_type momentum;

		const ErrFn err_fn;
		bool multi_thread;
		
		template <class Input>
		void feedforward(const Input &input)
		{
			size_t l = 0;
			for (auto &layer: ann) {
				if (l) layer->feedforward(act[l-1].cbegin(), act[l].begin());
				else layer->feedforward(input.cbegin(), act[l].begin());
				for (auto& v : act[l])
					assert(!isnan(v));
				++l;
			}

		}
 		template <class Output>
		data_type compute_error(const Output &target) // mean_sq_err
		{
			switch (err_fn) {
				case ErrFn::SquareError: 
					return inner_product(target.cbegin(), target.cend(), output_act.cbegin(), 0.0, plus<data_type>(), 
							[](data_type t, data_type o){ return (t-o)*(t-o);}) / (data_type)target.size();
				case ErrFn::CrossEntropy:
					return inner_product(target.cbegin(), target.cend(), output_act.cbegin(), 0.0, plus<data_type>(), 
							[](data_type t, data_type o){ return -t*log(o > 0 ? o : numeric_limits<data_type>::epsilon());});
			}
		}
		// d(E) same for sq err and log cross entropy, provided d(o)/d(net) = 1
		data_type compute_error_derived(const data_type o, const data_type t) { return (o-t); }

		template <class Input, class Output>
		void backpropagate(const Input &input, const Output &target)
		{
			typename ANN::layer_ptr next_layer = nullptr;
			size_t l = ann.size() - 1;
			for (auto &layer: ann.rev()){
				for (size_t j = 0; j < layer->size(); ++j) {
					data_type neuron_act = act[l][j];
					data_type d_act = layer->d_act_fn(neuron_act);
					if (l == ann.size() - 1) slope[l][j] = compute_error_derived(neuron_act, target[j]) * d_act;
					else slope[l][j] = inner_product(slope[l+1].begin(), slope[l+1].end(), next_layer->w_col_iter(j), 0.0) * d_act;
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
		data_type compute_batch_error(size_t sz) { 
			switch (err_fn) {
				case ErrFn::SquareError:  return sqrt(sum_err / sz);
				case ErrFn::CrossEntropy: return sum_err / sz;
			}
		}
		
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
			if (x < 0.000000000001) return 0; // default zero tolerance
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
		void apply_dw_rprop2( 
			V3D_ptr& prev_gradient_w, 
			V2D_ptr& prev_gradient_b, 
			data_type delta_max) /* rprop; used only in batch learning */
		{

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
						auto& delta_w = (*prev_dw)[l][j][i];
						if (change > 0) { // same sign
							//cout<<"same sign\n";
							delta_w = min(delta_w * eta_plus, delta_max); // update delta
							data_type dw = sign((*partial_dw)[l][j][i]) * delta_w;
							w_ij -= dw; // update weight
							prev_gradient_w_ = cur_gradient_w;
						} else if (change < 0) { // change sign: last delta was too big
							//w_ij -= (*delta_weight)[l][j][i]; // revert weight
							delta_w = max(delta_w * eta_minus, delta_min); // update delta
							prev_gradient_w_ = 0; // set previous gradient to zero so that there will be no adjustment to delta next round
						} else { // zero: no change in delta
							data_type dw = sign((*partial_dw)[l][j][i]) * delta_w;
							w_ij -= dw; // update weight
							prev_gradient_w_ = cur_gradient_w;
						}
						++i;
					}

					auto& prev_gradient_b_ = (*prev_gradient_b)[l][j];
					data_type cur_gradient_b = (*partial_db)[l][j];
					auto& delta_b = (*prev_db)[l][j];
					if (prev_gradient_b_ * cur_gradient_b > 0) {
						delta_b = min(delta_b * eta_plus, delta_max);
						data_type db = (-1) * sign((*partial_db)[l][j]) * delta_b;
						*b_j += db;
						prev_gradient_b_ = (*partial_db)[l][j];
					} else if (prev_gradient_b_ * cur_gradient_b < 0) {
						delta_b = max(delta_b * eta_minus, delta_min);
						prev_gradient_b_ = 0;
					} else {
						data_type db = sign((*partial_db)[l][j]) * delta_b;
						*b_j -= db;
						prev_gradient_b_ = (*partial_db)[l][j];
					}
					++j, ++b_j;
				}
				++l;
			}
		}
		void apply_dw_adagrad() {
			/*
			https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/

			autocorr = .95 #for example
			master_stepsize = 1e-2 #for example
			fudge_factor = 1e-6 #for numerical stability
			historical_grad = 0
			w = randn #initialize w
			while not converged:
			E,grad = computeGrad(w)
			if historical_grad == 0:
			historical_grad = g^2
			else:
			historical_grad = autocorr*historical_grad + (1-autocorr)*g^2
			adjusted_grad = grad / (fudge_factor + sqrt(historical_grad))
			w = w - master_stepsize*adjusted_grad
			*/
			data_type autocorr = 0.95, fudge_factor = 1e-6, master_stepsize = 1e-2;
			size_t l = 0;
			for (auto &layer: ann){
				size_t j = 0;
				auto b_j = layer->bias_begin();
				for (auto &w_j: *layer){
					size_t i = 0;
					for (auto & w_ij: w_j){
						auto& grad_w = (*partial_dw)[l][j][i]; // new gradient
						auto& historical_grad_w = (*prev_dw)[l][j][i]; // historical gradient
						if (historical_grad_w > -0.0000001 && historical_grad_w < 0.0000001) {
							historical_grad_w = grad_w * grad_w;
						} else {
							historical_grad_w = autocorr * historical_grad_w + (1-autocorr)*grad_w*grad_w;
						}
						data_type adjusted_grad_w = grad_w / (fudge_factor + sqrt(historical_grad_w)); //
						w_ij -= master_stepsize * adjusted_grad_w;
						++i;
					}
					auto& grad_d = (*partial_db)[l][j];
					auto& historical_grad_b = (*prev_db)[l][j];
					if (historical_grad_b > -0.0000001 && historical_grad_b < 0.0000001) {
						historical_grad_b = grad_d * grad_d;
					} else {
						historical_grad_b = autocorr * historical_grad_b + (1-autocorr)*grad_d*grad_d;
					}
					data_type adjusted_grad_d = grad_d / (fudge_factor + sqrt(historical_grad_b));
					*b_j -= master_stepsize * adjusted_grad_d;
					++j, ++b_j;
				}
				++l;
			}
		}
		/* implementation based on original paper */
		void apply_dw_quickprop(size_t n_samples, V3D_ptr& prev_gradient_w, V2D_ptr& prev_gradient_b) {
			data_type miu = 1.75; // maximum growth factor
			data_type epsilon = learning_rate / n_samples;
			data_type decay = -0.0001;

			size_t l = 0;
			for (auto &layer: ann){
				size_t j = 0;
				auto b_j = layer->bias_begin();
				for (auto &w_j: *layer){
					size_t i = 0;
					for (auto & w_ij: w_j){
						auto& prev_gradient_w_ = (*prev_gradient_w)[l][j][i]; // previous weight slope
						auto& dw_ = (*prev_dw)[l][j][i]; // previous delta w
						data_type cur_gradient_w = (*partial_dw)[l][j][i] + decay * w_ij; // current slope
						//cout << cur_gradient_w << "\n";
						int change = sign(prev_gradient_w_ * cur_gradient_w);
						//if (prev_gradient_w_ == cur_gradient_w) cout << "wo ca!!!" << cur_gradient_w <<"\n";

						// if (prev_gradient_w_ == cur_gradient_w) {
						// 	dw_ = epsilon * cur_gradient_w;
						// } else {
							// data_type factor = min(cur_gradient_w / (prev_gradient_w_ - cur_gradient_w), miu);
						data_type factor = 0;
						if (fabs(prev_gradient_w_ - cur_gradient_w) < 0.00000001) {
							factor = miu;
						} else {
							factor = cur_gradient_w / (prev_gradient_w_ - cur_gradient_w);
						}
						if (change < 0) { // current slope is opposite in sign from previous slope			
							dw_ = factor * dw_;
						} else {
							dw_ = factor * dw_ + epsilon * cur_gradient_w;
						}
						
						w_ij += dw_; // update weight
						prev_gradient_w_ = cur_gradient_w;
						++i;
					}

					auto& prev_gradient_b_ = (*prev_gradient_b)[l][j]; // previous bias slope
					auto& db_ = (*prev_db)[l][j]; // previous delta b
					data_type cur_gradient_b = (*partial_db)[l][j] + decay * (*b_j);
					int change = sign(prev_gradient_b_ * cur_gradient_b);
					data_type factor = 0;
						if (fabs(prev_gradient_b_ - cur_gradient_b) < 0.00000001) {
							factor = miu;
						} else {
							factor = cur_gradient_b / (prev_gradient_b_ - cur_gradient_b);
						}

					if (change < 0) {
						db_ = factor * db_;
					} else {
						db_ = factor * db_ + epsilon * cur_gradient_b;
					}
					*b_j += db_;
					prev_gradient_b_ = cur_gradient_b;
					++j, ++b_j;
				}
				++l;
			}
		}
		/* implementation based on FANN library */
		void apply_dw_quickprop2(size_t n_samples, V3D_ptr& prev_gradient_w, V2D_ptr& prev_gradient_b) {
			data_type miu = 1.75; // maximum growth factor
			data_type epsilon = learning_rate / n_samples;
			data_type decay = -0.0001;
			data_type shrink_factor = miu / (1.0 + miu);

			size_t l = 0;
			for (auto &layer: ann){
				size_t j = 0;
				auto b_j = layer->bias_begin();
				for (auto &w_j: *layer){
					size_t i = 0;
					for (auto & w_ij: w_j){
						auto& prev_gradient_w_ = (*prev_gradient_w)[l][j][i]; // previous weight slope
						auto& dw_ = (*prev_dw)[l][j][i]; // previous delta w
						data_type cur_gradient_w = (*partial_dw)[l][j][i] + decay * w_ij; // current slope
						data_type next_step = 0.0;
						if (dw_ > 0.001) {
							if (cur_gradient_w > 0.0) {
								next_step += epsilon * cur_gradient_w;
							}

							if (cur_gradient_w > (shrink_factor * prev_gradient_w_)) {
								next_step += miu * dw_;
							} else {
								next_step += dw_ * cur_gradient_w / (prev_gradient_w_ - cur_gradient_w);
							}
						} else if (dw_ < -0.001) {
							if (cur_gradient_w < 0.0) {
								next_step += epsilon * cur_gradient_w;
							}

							if (cur_gradient_w < (shrink_factor * prev_gradient_w_)) {
								next_step += miu * dw_;
							} else {
								next_step += dw_ * cur_gradient_w / (prev_gradient_w_ - cur_gradient_w);
							}
						} else {
							next_step += epsilon * cur_gradient_w;
						}
						
						dw_ = next_step;
						w_ij += dw_; // update weight
						prev_gradient_w_ = cur_gradient_w;
						++i;
					}

					auto& prev_gradient_b_ = (*prev_gradient_b)[l][j]; // previous bias slope
					auto& db_ = (*prev_db)[l][j]; // previous delta b
					data_type cur_gradient_b = (*partial_db)[l][j] + decay * (*b_j);
					data_type next_step = 0.0;
					if (db_ > 0.001) {
						if (cur_gradient_b > 0.0) {
							next_step += epsilon * cur_gradient_b;
						}

						if (cur_gradient_b > (shrink_factor * prev_gradient_b_)) {
							next_step += miu * db_;
						} else {
							next_step += db_ * cur_gradient_b / (prev_gradient_b_ - cur_gradient_b);
						}
					} else if (db_ < -0.001) {
						if (cur_gradient_b < 0.0) {
							next_step += epsilon * cur_gradient_b;
						}

						if (cur_gradient_b < (shrink_factor * prev_gradient_b_)) {
							next_step += miu * db_;
						} else {
							next_step += db_ * cur_gradient_b / (prev_gradient_b_ - cur_gradient_b);
						}
					} else {
						next_step += epsilon * cur_gradient_b;
					}

					db_ = next_step;
					*b_j += db_;
					prev_gradient_b_ = cur_gradient_b;
					++j, ++b_j;
				}
				++l;
			}
		}
		void print_progress(size_t i, size_t epochs, size_t p = 100) {
			size_t segment = epochs / p;
			if (segment && i % segment && i != epochs) return;
			double percentage = 100.0 * i / epochs;			
			cout<<percentage<<"% epoch "<<i<<" training error: "<<compute_batch_error(data_range.size())<<"\n";
			if (test_range_ptr != nullptr) {
				sum_err = 0;
				for (auto &data : *test_range_ptr) {
					feedforward(data.first);
					sum_err += compute_error(data.second);
				}
				cout<<"        testing error: "<<compute_batch_error(test_range_ptr->size())<<"\n";
			}
			if (i == epochs) ann.print();
		}
		// data_type train_epilogue() {
		// 	sum_err = 0;
		// 	for (auto &data : data_range){
		// 		feedforward(data.first);
		// 		cout<<"test---";
		// 		for( auto v :data.first) cout<<v<<" ";
		// 		cout<<"answer:";
		// 		for( auto v :output_act) cout<<v<<" ";
		// 			cout<<" should be:";
		// 		for( auto v :data.second) cout<<v<<" ";
		// 			cout<<"\n";
		// 		sum_err += compute_error(data.second);
		// 	}
		// 	ann.print();
		// 	return compute_batch_error();
		// }

		void fill_v2d(V2D &v2d, data_type val = 0) {for (auto &v1d:v2d) fill(v1d.begin(), v1d.end(), val); }
		void fill_v3d(V3D &v3d, data_type val = 0) {for (auto &v2d:v3d) fill_v2d(v2d, val); }

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
		Trainer(ANN &_ann, const DataRange &_data_range, data_type alpha = (data_type)0.01, 
			data_type beta = (data_type)0.9, const ErrFn _err_fn = ErrFn::SquareError)

		:	ann(_ann), act(ann.size()), output_act(act[ann.size()-1]), slope(ann.size()), 
			prev_dw(make_unique<V3D>(ann.size())), partial_dw(make_unique<V3D>(ann.size())),
			prev_db(make_unique<V2D>(ann.size())), partial_db(make_unique<V2D>(ann.size())),
			data_range((assert("Dataset must not be empty" && _data_range.size()), _data_range)),
			test_range_ptr(nullptr), learning_rate(alpha), momentum(beta), err_fn(_err_fn), multi_thread(false)
		{
			init_vectors(
				{&act, &slope, prev_db.get(), partial_db.get()},
				{prev_dw.get(), partial_dw.get()});
		}

		// construct multi-thread trainer;
		Trainer(bool placeholer, ANN &_ann, const DataRange &_data_range, const ErrFn _err_fn = ErrFn::SquareError)
		:	ann(_ann), act(ann.size()), output_act(act[ann.size()-1]), slope(ann.size()), 
			partial_dw(make_unique<V3D>(ann.size())), partial_db(make_unique<V2D>(ann.size())),
			data_range((assert("Dataset must not be empty" && _data_range.size()), _data_range)), 
			test_range_ptr(nullptr), err_fn(_err_fn), multi_thread(true)
		{
			(void)placeholer;
			init_vectors(
				{&act, &slope, partial_db.get()},
				{partial_dw.get()});
		}

		Trainer(const Trainer& rhs) = delete;
		Trainer(Trainer& rhs) = delete;

		void set_test_data(DataRange * _test_range_ptr) { test_range_ptr = _test_range_ptr; }

		void train_stochastic(size_t epochs, size_t p = 100)
		{
			assert("multi-thread trainer" && !multi_thread);
			fill_v3d(*prev_dw);
			fill_v2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i) {
				sum_err = 0;
				for (auto &data : data_range){
					fill_v3d(*partial_dw);
					fill_v2d(*partial_db);
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					apply_dw(1);
				}
				print_progress(i, epochs, p);
			}
			print_progress(epochs, epochs);
		}

		void train_minibatch(size_t epochs, size_t batch_sz, size_t p = 100)
		{
			assert("multi-thread trainer" && !multi_thread);
			fill_v3d(*prev_dw);
			fill_v2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i) {
				sum_err = 0;
				auto data_iter = data_range.begin();
				auto data_end = data_range.end();
				size_t cnt = 0;
				while (data_iter != data_end){
					fill_v3d(*partial_dw);
					fill_v2d(*partial_db);
					feedforward(data_iter->first);
					backpropagate(data_iter->first, data_iter->second);
					sum_err += compute_error(data_iter->second);					
					++data_iter, ++cnt;
					if (cnt == batch_sz || data_iter == data_end){
						apply_dw(cnt);
						cnt = 0;
					}
				}
				print_progress(i, epochs, p);
			}
			print_progress(epochs, epochs);
		}

		void train_batch(size_t epochs, size_t p = 100) // return final error
		{
			assert("multi-thread trainer" && !multi_thread);
			fill_v3d(*prev_dw);
			fill_v2d(*prev_db);
			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_v3d(*partial_dw);
				fill_v2d(*partial_db);
				size_t k = 0;
				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					++k;
					if (k%(data_range.size()/10) == 0)
						cout<<"                  "<<k<<" of "<<data_range.size()<<"\n";
				}
				print_progress(i, epochs, p);
				apply_dw(data_range.size());
			}
			print_progress(epochs, epochs);
		}

		void train_rprop(size_t epochs, data_type delta_initial = 0.1, data_type delta_max = 50.0, size_t p = 100) // rprop algorithmn based on original paper
		{
			assert("multi-thread trainer" && !multi_thread);

			fill_v3d(*prev_dw);
			fill_v2d(*prev_db);

			// rprop specific
			V3D_ptr prev_gradient_w = make_unique<V3D>(ann.size()); // dE/dw_ij(t-1)
			V2D_ptr prev_gradient_b = make_unique<V2D>(ann.size()); // dE/db_i(t-1)
			V3D_ptr delta_weight = make_unique<V3D>(ann.size()); // delta_ij for weight
			V2D_ptr delta_bias = make_unique<V2D>(ann.size()); // delta_ij for bias
			init_vectors(
				{prev_gradient_b.get(), delta_bias.get()},
				{prev_gradient_w.get(), delta_weight.get()});

			fill_v3d(*prev_gradient_w);
			fill_v2d(*prev_gradient_b); 
			fill_v3d(*delta_weight, delta_initial);
			fill_v2d(*delta_bias, delta_initial);

			//cout<<(*delta_weight)[0][0][0];
			//return 0.1;

			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_v3d(*partial_dw);
				fill_v2d(*partial_db);
				int k = 0;
				//cout<<"ha: " <<(*delta_weight)[1][1][1]<<"\n";
				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					k++;
				}
				print_progress(i, epochs, p);
				apply_dw_rprop(delta_weight, prev_gradient_w, delta_bias, prev_gradient_b, delta_max);
				//if (i == 2) return 0;
			}
			print_progress(epochs, epochs);
		}
		void train_rprop2(size_t epochs, data_type delta_initial = 0.1, data_type delta_max = 50.0, size_t p = 100) // rprop algorithmn based on original paper
		{
			assert("multi-thread trainer" && !multi_thread);

			//fill_v3d(*prev_dw);
			//fill_v2d(*prev_db);

			// rprop specific
			V3D_ptr prev_gradient_w = make_unique<V3D>(ann.size()); // dE/dw_ij(t-1)
			V2D_ptr prev_gradient_b = make_unique<V2D>(ann.size()); // dE/db_i(t-1)
			//V3D_ptr delta_weight = make_unique<V3D>(ann.size()); // delta_ij for weight
			//V2D_ptr delta_bias = make_unique<V2D>(ann.size()); // delta_ij for bias
			init_vectors(
				{prev_gradient_b.get()},
				{prev_gradient_w.get()});

			fill_v3d(*prev_gradient_w);
			fill_v2d(*prev_gradient_b); 
			fill_v3d(*prev_dw, delta_initial);
			fill_v2d(*prev_db, delta_initial);

			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_v3d(*partial_dw);
				fill_v2d(*partial_db);
				int k = 0;
				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					k++;
				}
				print_progress(i, epochs, p);
				apply_dw_rprop2(prev_gradient_w, prev_gradient_b, delta_max);
			}
			print_progress(epochs, epochs);
		}

		void train_adagrad(size_t epochs, size_t p = 100) // adagrad algorithm
		{
			assert("multi-thread trainer" && !multi_thread);

			fill_v3d(*prev_dw);
			fill_v2d(*prev_db);

			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_v3d(*partial_dw);
				fill_v2d(*partial_db);
				int k = 0;

				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					k++;
				}
				print_progress(i, epochs, p);
				apply_dw_adagrad();
			}
			print_progress(epochs, epochs);
		}

		void train_quickprop(size_t epochs, size_t p = 100) // quickprop algorithm
		{
			assert("multi-thread trainer" && !multi_thread);

			fill_v3d(*prev_dw);
			fill_v2d(*prev_db);

			// rprop specific
			V3D_ptr prev_gradient_w = make_unique<V3D>(ann.size()); // dE/dw_ij(t-1)
			V2D_ptr prev_gradient_b = make_unique<V2D>(ann.size()); // dE/db_i(t-1)

			init_vectors(
				{prev_gradient_b.get()},
				{prev_gradient_w.get()});

			fill_v3d(*prev_gradient_w);
			fill_v2d(*prev_gradient_b); 

			for (size_t i = 0; i < epochs; ++i){
				sum_err = 0;
				fill_v3d(*partial_dw);
				fill_v2d(*partial_db);
				int k = 0;

				for (auto &data : data_range){
					feedforward(data.first);
					backpropagate(data.first, data.second);
					sum_err += compute_error(data.second);
					k++;
				}
				print_progress(i, epochs, p);
				apply_dw_quickprop(data_range.size(), prev_gradient_w, prev_gradient_b);
			}
			print_progress(epochs, epochs);
		}



		// return sum of errors
		data_type train_multi_thread()
		{
			assert("multi-thread trainer" && multi_thread);
			fill_v3d(*partial_dw);
			fill_v2d(*partial_db);
			sum_err = 0;
			for (auto &data : data_range){
				feedforward(data.first);
				backpropagate(data.first, data.second);
				sum_err += compute_error(data.second);
			}
			return sum_err;
		}
		data_type get_error_multi_thread(bool run_test = false) {
			sum_err = 0;
			if (run_test && test_range_ptr != nullptr)
				for (auto &data : *test_range_ptr){
					feedforward(data.first);
					sum_err += compute_error(data.second);
				}
			else
				for (auto &data : data_range){
					feedforward(data.first);
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
}

#endif /*NN_TRAINER_H*/
