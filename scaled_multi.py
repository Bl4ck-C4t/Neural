from multi_layer import *

mn = 1
mx = 1

def normalize_all(data):
    global mx
    global mn
    mn = min(data)
    mx = max(data)
    return [(x - mn) / (mx - mn) for x in data]


if __name__ == "__main__":
    FNAME = "scaled_data.nn"
    DATASET_LEN = 20
    ITER = 60000
    # Seed the random number generator
    if path.isfile(f"./{FNAME}"):
        neural_network = NeuralNetwork.load(FNAME)
    else:
        random.seed(1)
        # layers = [500, 200, 100, 50, 25, 10, 5]
        layers = [100, 20, 5]
        # Combine the layers to create a neural network
        # neural_network = NeuralNetwork([layer1, layer2, layer3, layer4])
        neural_network = create_network(3, 1, layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    inputs = [[NeuralNetwork.normalize(random.randint(0, 100), 0, 100)
               for i in range(3)] for x in range(DATASET_LEN)]
    training_set_inputs = array(inputs)
    training_set_outputs = array([[NeuralNetwork.normalize(sum(x),0, 300) for x in inputs]]).T
    # training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    # training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, ITER)

    print("Stage 2) New synaptic weights after training: ")
    # neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [3, 4, 9] -> ?(16): ")
    new = [3, 4, 9]
    actual = sum(new)
    outputs = list(neural_network.work(array(new)))
    val = NeuralNetwork.denormalize(outputs[-1], 0, 300)
    print(val)
    diff = abs(actual - val)
    print(f"Missed by {actual/diff*100}%")
    # print([abs(100 - b % round(b) * 100) for b in outputs])
    # neural_network.save(FNAME)
