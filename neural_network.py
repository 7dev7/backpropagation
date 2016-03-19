import numpy as np
import dataset as ds


class NNetwork:
    input_layer_size = 25
    hidden_layer_size = 7
    output_layer_size = 10

    def __init__(self):
        np.random.seed(1)
        self.synapse0 = 2 * np.random.random((self.input_layer_size, self.hidden_layer_size)) - 1
        self.synapse1 = 2 * np.random.random((self.hidden_layer_size, self.output_layer_size)) - 1

        self.hidden_layer = np.zeros(self.hidden_layer_size)
        self.output_layer = np.zeros(self.output_layer_size)

        self.out_neuron_errors = np.zeros(self.output_layer_size)
        self.hidden_neuron_errors = np.zeros(self.hidden_layer_size)

        self.start_learning()

    @staticmethod
    def __get_weighted_sum(layer, synapse, num_of_neuron: int) -> float:
        res = 0
        for i in range(len(synapse)):
            res += layer[i] * synapse[i][num_of_neuron]
        return res

    @staticmethod
    def __sigmoid(x: float, is_derivative=False) -> float:
        if is_derivative is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def _study(self, in_data, out_data):
        for i in range(self.hidden_layer_size):
            w_sum = self.__get_weighted_sum(in_data, self.synapse0, i)
            activated = self.__sigmoid(w_sum)
            self.hidden_layer[i] = activated

        activated = 0

        for i in range(self.output_layer_size):
            w_sum = self.__get_weighted_sum(self.hidden_layer, self.synapse1, i)
            activated = self.__sigmoid(w_sum)
            self.output_layer[i] = activated

        # Back propagation

        for i in range(self.output_layer_size):
            self.out_neuron_errors[i] = (out_data[i] - self.output_layer[i]) * self.__sigmoid(self.output_layer[i],
                                                                                              True)

        for i in range(self.hidden_layer_size):
            current_error = 0
            for k in range(self.output_layer_size):
                current_error += self.out_neuron_errors[k] * self.synapse1[i][k]
            self.hidden_neuron_errors[i] = current_error * self.__sigmoid(self.hidden_layer[i], True)

        # Change synapses weights

        for i in range(self.output_layer_size):
            for j in range(self.hidden_layer_size):
                delta = self.out_neuron_errors[i] * self.hidden_layer[j]
                self.synapse1[j][i] += delta

        for i in range(self.hidden_layer_size):
            for j in range(self.input_layer_size):
                delta = self.hidden_neuron_errors[i] * in_data[j]
                self.synapse0[j][i] += delta

    def start_learning(self):
        for i in range(100):
            for j in range(100):
                if j == 0:
                    self._study(ds.ZERO, ds.expected_zero)
                elif j == 1:
                    self._study(ds.ONE, ds.expected_one)
                elif j == 2:
                    self._study(ds.TWO, ds.expected_two)
                elif j == 3:
                    self._study(ds.THREE, ds.expected_three)
                elif j == 4:
                    self._study(ds.FOUR, ds.expected_four)
                elif j == 5:
                    self._study(ds.FIVE, ds.expected_five)
                elif j == 6:
                    self._study(ds.SIX, ds.expected_six)
                elif j == 7:
                    self._study(ds.SEVEN, ds.expected_seven)
                elif j == 8:
                    self._study(ds.EIGHT, ds.expected_eight)
                elif j == 9:
                    self._study(ds.NINE, ds.expected_nine)

    def recognize(self, data):
        for i in range(self.hidden_layer_size):
            w_sum = self.__get_weighted_sum(data, self.synapse0, i)
            activated = self.__sigmoid(w_sum)
            self.hidden_layer[i] = activated

        for i in range(self.output_layer_size):
            w_sum = self.__get_weighted_sum(self.hidden_layer, self.synapse1, i)
            activated = self.__sigmoid(w_sum)
            self.output_layer[i] = activated

        for i in range(len(self.output_layer)):
            print(i, ": ", self.output_layer[i] * 1000)
