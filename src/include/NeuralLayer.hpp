#ifndef NN_NEURALLAYER_HPP
#define NN_NEURALLAYER_HPP

#include <Utility.hpp>

namespace NN
{
	using namespace std;

	enum class ActFn
	{
		Input, // not used
		Sigmoid,
		Tanh,
		ReLU,
		Softplus,
		Linear,
		Softmax
	};
	template <class Ty>
	class NeuralLayer
	{
	private:
		size_t in_sz;
		size_t sz;
		vector<vector<Ty>> w; // weight
		vector<Ty> b; // bias
		const ActFn act;
		function<Ty(Ty)> act_fn;
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
		function<Ty(Ty)> d_act_fn;
		void print(){
			switch (act){
				case ActFn::Input: break;
				case ActFn::Sigmoid: cout<<"Sigmoid"; break;
				case ActFn::Tanh: cout<<"Tanh"; break;
				case ActFn::ReLU: cout<<"ReLU"; break;
				case ActFn::Softplus: cout<<"Softplus"; break;
				case ActFn::Linear: cout<<"Linear"; break;
				case ActFn::Softmax: cout<<"Softmax"; break;
			}
			cout<<" Layer "<<in_sz<<" input, "<<sz<<" output.\n";
			// cout<<"Weight:\n";
			// for (auto &w_j : w){
			// 	for (auto w_ij: w_j) cout<<w_ij<<" ";
			// 	cout<<"\n";
			// }
			// cout<<"Bias:\n";
			// for (auto b_j : b) cout<<b_j<<" ";
			// cout<<"\n";
		}
		typedef Ty data_type;
		NeuralLayer(size_t in, size_t out, ActFn _act = ActFn::Sigmoid, bool init_random_w = true)
		: in_sz(in), sz(out), w(out, vector<Ty>(in)), b(out), act(_act)
		{
			switch (act){
				case ActFn::Input: break;
				case ActFn::Sigmoid: 
					act_fn = [](Ty x){return 1.0/(1 + exp(-x));};
					d_act_fn = [](Ty fx){return (Ty)fx * (1 - fx);}; break;
				case ActFn::Tanh:
					act_fn = [](Ty x){return tanh(x);};
					d_act_fn = [](Ty fx){return 1 - fx * fx;}; break;
				case ActFn::ReLU:
					act_fn = [](Ty x){return max(x, (Ty)0);};
					d_act_fn = [](Ty fx){return fx > 0 ? 1: 0;}; break;
				case ActFn::Softplus:
					act_fn = [](Ty x){return log1p(exp(x));};
					d_act_fn = [](Ty fx){return 1 - exp(-fx);}; break;
				case ActFn::Linear:
					act_fn = [](Ty x){return x;};
					d_act_fn = [](Ty fx){return (void)fx, 1;}; break; // tackle -Wunused-parameter
				case ActFn::Softmax:
					// no acf_fn, fx depend on all x_j
					d_act_fn = [](Ty fx){return (void)fx, 1;}; break; // tackle -Wunused-parameter
					// trick for softmax output layer d(cross entropy)/dWji = (Act[j] - Target[j]) * Act[i]
			}
			
			if (!init_random_w) return;
			
			Ty r = sqrt((Ty)6.0/(in + out));
			mt19937 gen((random_device()()));
			// mt19937 gen;
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
			vector<Ty> net(sz);
			transform(w.cbegin(), w.cend(), b.cbegin(), net.begin(), 
				[=](auto&w_i, auto&b_i){ 
					return b_i + inner_product(w_i.cbegin(), w_i.cend(), in, (Ty)0.0); });

			if (act == ActFn::Softmax) {
				// https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
				Ty net_max = *max_element(net.cbegin(), net.cend());
				Ty deno = 0;
				for (Ty& net_j : net) {
					net_j -= net_max;
					deno += exp(net_j);
				}
				deno = log(deno);
				transform(net.cbegin(), net.cend(), out, [=](Ty net_j){ return exp(net_j - deno);});
			} else transform(net.cbegin(), net.cend(), out, act_fn);
		}

		column_iterator w_col_iter(size_t idx) { return column_iterator(w.cbegin(), idx);}
	};
}

#endif /*NN_NEURALLAYER_HPP*/
