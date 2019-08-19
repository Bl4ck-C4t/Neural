from numpy import exp, array, dot, random
from typing import *
from collections import namedtuple
import datetime
import pickle
from os import path
import matplotlib.pyplot as plt


# random.seed(1)

# def time_info(iteration):
#     def my_decorator(func):
#         def wrapper(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
#             print("Something is happening before the function is called." + str(n))
#             func()
#             curr = datetime.datetime.now()
#             func(training_set_inputs, training_set_outputs, 1)
#             delta = datetime.datetime.now() - curr
#
#         return wrapper
#     return my_decorator

def time_info(func):
    def wrapper(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        # data = []
        # for x in range(30):
        #     curr = datetime.datetime.now()
        #     func(self, training_set_inputs, training_set_outputs, 1)
        #     delta = datetime.datetime.now() - curr
        #     data.append(delta.microseconds)
        # c = max(data)*number_of_training_iterations
        procent = 30/6000
        sample = int(number_of_training_iterations * procent)
        print("Calculating approximate time...")
        curr = datetime.datetime.now()
        func(self, training_set_inputs, training_set_outputs, sample)
        delta = datetime.datetime.now() - curr
        c = delta.microseconds/sample*number_of_training_iterations
        print(f"Approximate time for all iterations: {datetime.timedelta(0, 0, c)}")
        curr = datetime.datetime.now()
        func(self, training_set_inputs, training_set_outputs, number_of_training_iterations)
        print(f"Completed for {datetime.datetime.now() - curr}")

    return wrapper


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
    @time_info
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        # data = []
        for iteration in range(number_of_training_iterations):
            # curr = datetime.datetime.now()
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
            # data.append(datetime.datetime.now() - curr)
            # data[-1] = data[-1].microseconds

        # plt.plot(data, "-o")
        # plt.ylabel('data')
        # plt.show()
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

    def work(self, inputs):
        return self.think(inputs)[-1]

    def save(self, fname="data.nn"):
        with open(fname, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname="data.nn"):
        with open(fname, "rb") as f:
            return pickle.load(f)

    # The neural network prints its weights
    def print_weights(self):
        for layer in range(len(self.layers)):
            print(f"    Layer {layer} ({self.layers[layer].synaptic_weights.shape[1]} neurons, "
                  f"each with {self.layers[layer].synaptic_weights.shape[0]} inputs): ")
            print(self.layers[layer].synaptic_weights)


# [100,20,5]
def create_network(inputs, outputs, neurons_per_layer: List[int]) -> NeuralNetwork:
    layers = []
    inp_layer = NeuronLayer(neurons_per_layer[0], inputs)
    inp_layer.name = "Layer 0"
    out_layer = NeuronLayer(outputs, neurons_per_layer[-1])
    layers = [inp_layer]
    for neurons in neurons_per_layer[1:]:
        layers.append(NeuronLayer(neurons, layers[-1].synaptic_weights.shape[1]))
    layers.append(out_layer)
    return NeuralNetwork(layers)


if __name__ == "__main__":
    # Seed the random number generator
    if path.isfile("./data.nn"):
        neural_network = NeuralNetwork.load()
    else:
        random.seed(1)


        # Create layer 1 (4 neurons, each with 3 inputs)
        layer1 = NeuronLayer(100, 3)

        # Create layer 2 (a single neuron with 4 inputs)
        layer2 = NeuronLayer(20, 100)
        layer3 = NeuronLayer(5, 20)
        layer4 = NeuronLayer(1, 5)

        layers = [100, 20, 5]

        # Combine the layers to create a neural network
        # neural_network = NeuralNetwork([layer1, layer2, layer3, layer4])
        neural_network = create_network(3, 1, layers)

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
    # neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    outputs = neural_network.work(array([1, 1, 0]))
    print(outputs)
    neural_network.save()
