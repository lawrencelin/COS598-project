#ifndef NN_NEURALNETWORK_H
#define NN_NEURALNETWORK_H

#include <NeuralLayer.hpp>

namespace NN
{
	using namespace std;

	template <class Ty>
	class NeuralNetwork
	{
	private:
		// Not counting input layers: input x hidden x output = 2 layers.
		typedef unique_ptr<NeuralLayer<Ty> > layer_type; // using pointers to allow classes derived from NeuralLayer
		typedef typename vector<layer_type>::const_iterator layer_iter;
		vector<layer_type> layers;
		
		// For iterating the neural network backwards
		class reverse_adapter {
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
		NeuralNetwork(const initializer_list<size_t> l, const ActFn hidden_type = ActFn::Sigmoid, const ActFn output_type = ActFn::Linear)
		: layers((
			assert("Softmax is only for output layer" && hidden_type != ActFn::Softmax),
			assert("NeuralNetwork need at least one non-input layer!" && l.size() > 1),
			l.size() - 1))
		{
			size_t in, i = 0;
			for (size_t out : l) {
				if (i) layers[i-1] = make_unique<NeuralLayer<Ty>>(in, out, 
					(i + 1 == l.size()) ? output_type : hidden_type);
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
}

#endif /*NN_NEURALNETWORK_H*/
