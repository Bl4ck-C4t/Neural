from numpy import exp, array, random, dot, random
from typing import *
from collections import namedtuple
# random.seed(1)

error = namedtuple("error", ["error", "layer"])


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.name = ""
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

class NeuralNetwork():
    def __init__(self, layers: List[NeuronLayer]):
        for i, layer in zip(range(len(layers)), layers):
            layer.name = f"Layer {i}"
        self.layers = layers

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            outputs = self.think(training_set_inputs)[::-1]
            last_error = training_set_outputs - outputs[0]
            last_delta = last_error * self.__sigmoid_derivative(outputs[0])

            errors_and_layers: List[error] = [error(last_error, self.layers[-1])]
            deltas = [last_delta]
            for layer, output in zip(self.layers[::-1][1:], outputs[1:]):
                current_error = deltas[-1].dot(errors_and_layers[-1].layer.synaptic_weights.T)
                current_delta = current_error * self.__sigmoid_derivative(output)
                errors_and_layers.append(error(current_error, layer))
                deltas.append(current_delta)
            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            # layer3_error = training_set_outputs - output_from_layer_3
            # layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)

            # Calculate how much to adjust the weights by
            outputs = outputs[::-1]
            del outputs[-1]
            outputs.insert(0, training_set_inputs)

            # layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            # layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            # layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)
            adjustments = [out.T.dot(delta) for out, delta in zip(outputs, deltas[::-1])]
            for layer, adjust in zip(self.layers, adjustments):
                layer.synaptic_weights += adjust
            # Adjust the weights.


    # The neural network thinks.
    def think(self, inputs):
        # output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        # output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        # output_from_layer3 = self.__sigmoid(dot(output_from_layer2, self.layer3.synaptic_weights))
        outputs = [self.__sigmoid(dot(inputs, self.layers[0].synaptic_weights))]
        for layer in self.layers[1:]:
            outputs.append(self.__sigmoid(dot(outputs[-1], layer.synaptic_weights)))
        return outputs

    # The neural network prints its weights
    def print_weights(self):
        for layer in range(len(self.layers)):
            print(f"    Layer {layer} ({self.layers[layer].synaptic_weights.shape[1]} neurons, "
                  f"each with {self.layers[layer].synaptic_weights.shape[0]} inputs): ")
            print(self.layers[layer].synaptic_weights)


if __name__ == "__main__":
    # Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)


    # Combine the layers to create a neural network
    neural_network = NeuralNetwork([layer1, layer2])

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    hidden_states = neural_network.think(array([1, 1, 0]))
    print(hidden_states[-1])
