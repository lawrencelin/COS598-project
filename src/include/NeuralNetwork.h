#ifndef __NEURALNETWORK_H
#define __NEURALNETWORK_H

class NeuralNetwork
{
public:
	NeuralNetwork();
	~NeuralNetwork();
	
};

void train(){
	// d(E)/d(w)  = d(E)/d(phi) * d(phi)/d(net) * d(net)/d(w)

	// mini-batch to threads
	feedforward(input); // compute activations (phi) and (phi)'
	// mini-batch to threads
	backpropagate(output); // compute d(E)/d(phi) = sum(d(E)/d(phi) * w) and update delta(w)

	// back to main thread
	updateweights(); // w += sum(delta(w)) * learning_rate / n OR rprop / quickprop


}

#endif /*__NEURALNETWORK_H*/