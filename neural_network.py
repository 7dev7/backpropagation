import numpy as np
import dataset as ds


def sigmoid(x, is_derivative=False) -> float:
    if is_derivative is True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def get_weighted_sum(layer, synapse, num_of_neuron: int) -> float:
    res = 0
    for i in range(len(synapse)):
        res += layer[i] * synapse[i][num_of_neuron]
    return res


input_layer_size = 25
hidden_layer_size = 7
output_layer_size = 10


np.random.seed(1)
synapse0 = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
synapse1 = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1

hidden_layer = np.zeros(hidden_layer_size)
output_layer = np.zeros(output_layer_size)

out_neuron_errors = np.zeros(output_layer_size)
hidden_neuron_errors = np.zeros(hidden_layer_size)


def study(in_data, out_data):
    for i in range(hidden_layer_size):
        w_sum = get_weighted_sum(in_data, synapse0, i)
        activated = sigmoid(w_sum)
        hidden_layer[i] = activated

    activated = 0

    for i in range(output_layer_size):
        w_sum = get_weighted_sum(hidden_layer, synapse1, i)
        activated = sigmoid(w_sum)
        output_layer[i] = activated

    # Back propagation

    for i in range(output_layer_size):
        out_neuron_errors[i] = (out_data[i] - output_layer[i]) * sigmoid(output_layer[i], True)

    for i in range(hidden_layer_size):
        current_error = 0
        for k in range(output_layer_size):
            current_error += out_neuron_errors[k] * synapse1[i][k]
        hidden_neuron_errors[i] = current_error * sigmoid(hidden_layer[i], True)

    for i in range(output_layer_size):
        for j in range(hidden_layer_size):
            delta = out_neuron_errors[i] * hidden_layer[j]
            synapse1[j][i] += delta

    for i in range(hidden_layer_size):
        for j in range(input_layer_size):
            delta = hidden_neuron_errors[i] * in_data[j]
            synapse0[j][i] += delta


def start_learning():
    for i in range(100):
        for j in range(100):
            if j == 0:
                study(ds.ZERO, ds.expected_zero)
            elif j == 1:
                study(ds.ONE, ds.expected_one)
            elif j == 2:
                study(ds.TWO, ds.expected_two)
            elif j == 3:
                study(ds.THREE, ds.expected_three)
            elif j == 4:
                study(ds.FOUR, ds.expected_four)
            elif j == 5:
                study(ds.FIVE, ds.expected_five)
            elif j == 6:
                study(ds.SIX, ds.expected_six)
            elif j == 7:
                study(ds.SEVEN, ds.expected_seven)
            elif j == 8:
                study(ds.EIGHT, ds.expected_eight)
            elif j == 9:
                study(ds.NINE, ds.expected_nine)


def start(data):
    for i in range(hidden_layer_size):
        w_sum = get_weighted_sum(data, synapse0, i)
        activated = sigmoid(w_sum)
        hidden_layer[i] = activated

    for i in range(output_layer_size):
        w_sum = get_weighted_sum(hidden_layer, synapse1, i)
        activated = sigmoid(w_sum)
        output_layer[i] = activated

    for i in range(len(output_layer)):
        print(i, ": ", output_layer[i] * 1000)


def main():
    start_learning()

    print("Zero: ")
    start(ds.ZERO)
    print("----------------------")
    print("Two: ")
    start(ds.TWO)
    print("----------------------")
    print("Seven: ")
    start(ds.SEVEN)
    print("----------------------")
    print("Five: ")
    start(ds.FIVE)
    print("----------------------")
    print("Nine: ")
    start(ds.NINE)


if '__main__' == __name__:
    main()