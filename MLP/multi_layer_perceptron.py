#encoding:utf-8
import numpy as np
np.random.seed(20170430)

"""
Initialization, init weights and bias of each layer_i to layer_{i+1}

Forward-propagation, predict output of each layer for the input.

Back-propagation, compute gradient of each weight.

Update weights

"""


class MLP:

    def __init__(self,
                 n_input_unit=2,
                 n_output_unit=1,
                 n_hidden_units=[3],
                 n_epoch=5,
                 batch_size=5,
                 learning_rate=1e-2):
        """
        Specified Networks.

        n_layers: number of hidder layers
        weights^{l}: n_unit^{l-1} * n_unit^{l}
        bias^{l}: 1 * n_unit^{l}

        """

        self.n_input_unit = n_input_unit
        self.n_output_unit = n_output_unit
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layer = len(n_hidden_units)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.network = []

        # Hidden Layers
        for l in range(self.n_hidden_layer):
            # L hidden layers and 1 output layer, layer_{-1} is the input layer.
            # Weights of layer_{l} link layer_{l-1} and layer_{l}
            # Add bias in the weights, each bias for an output unit.

            layer = {}

            if l == 0:
                layer['weights'] = np.random.random((n_input_unit, n_hidden_units[l]))
            else:
                layer['weights'] = np.random.random((n_hidden_units[l-1], n_hidden_units[l]))

            layer['bias'] = np.random.random((1, n_hidden_units[l]))
            self.network.append(layer)

        # Output layer, one unit for regression
        layer = {}
        layer['weights'] = np.random.random((n_hidden_units[-1], n_output_unit))
        layer['bias'] = np.random.random((1, n_output_unit))
        self.network.append(layer)


    def summary(self):
        for i in range(self.n_hidden_layer + 1):
            layer = self.network[i]
            print('Layer-%s weights: %s\n, bias: %s\n, delta:%s\n, input:%s\n, output:%s\n'
                  % (i, layer['weights'], layer['bias'], layer['delta'], layer['input'], layer['output']))


    def fit(self, X, y):

        #print(self.network[0]['weights'], self.network[0]['bias'])
        #print(self.network[1]['weights'], self.network[1]['bias'])

        for epoch in range(self.n_epoch):

            for batch in range(int(len(X) / self.batch_size)):

                X_mini_batch = X[batch * self.batch_size : (batch+1) * self.batch_size]
                y_mini_batch = y[batch * self.batch_size : (batch+1) * self.batch_size]

                self._forward_propagation(X_mini_batch)
                self._back_propagation(X_mini_batch, y_mini_batch)
                self._update_weight(X_mini_batch, y_mini_batch)


            loss = self._compute_loss(X, y)
            print('Epoch %s, loss=%s' % (epoch, loss))

            '''
            print('--begin--')
            print('X=%s,y=%s' % (X_mini_batch, y_mini_batch))
            self.summary()
            print('--end--\n')
            '''



    def _compute_loss(self, X, y):

        pred = self.predict(X) # batch_size * n_output_unit

        batch_size = len(X)

        loss = 1. / batch_size * 1 / 2 * np.sum(np.subtract(pred, y) ** 2)
        return loss


    def _get_weighted_sum(self, X, weights, bias):
        """
        X: batch_size * n_unit^{l-1};  n_unit^{-1} = n_mini_batch
        weights: n_unit^{l-1} * n_unit^{l}
        bias: n_unit^{l}

        return: batch_size * n_unit^{l}
        """

        return np.dot(X, weights) + bias


    def _forward_propagation(self, X):
        """
        input:  batch_size * n_input_unit
        output: batch_size * n_output_unit
        """

        pre_layer_output = X

        for l in range(self.n_hidden_layer + 1):
            layer = self.network[l]
            layer_weights = layer['weights']
            layer_bias = layer['bias']

            layer_input = self._get_weighted_sum(pre_layer_output, layer_weights, layer_bias)
            layer_output = self._activate(layer_input)

            self.network[l]['input'] = layer_input
            self.network[l]['output'] = layer_output
            pre_layer_output = layer_output


    def _back_propagation(self, X, y):

        """
        X: batch_size * n_input_unit
        y: batch_size * n_output_unit

        delta: batch_size * n_unit^{l}

        derivative(X), same with X, batch_size * n_input_unit

        """

        for l in reversed(range(self.n_hidden_layer + 1)):

            layer = self.network[l]

            layer_output = layer['output'] # batch_size * n_unit^{l}
            derivative = self._derivative(layer_output)  # batch_size * n_unit^{l}

            if l == self.n_hidden_layer:
                error = np.subtract(layer_output, y) # batch_size * n_output_unit

                self.network[l]['delta'] = np.multiply(error, derivative) # batch_size * n_output_unit

            else:
                post_delta = self.network[l+1]['delta'] # batch_size * n_unit^{l+1}
                post_weights = self.network[l+1]['weights'] # n_unit^{l} * n_unit^{l+1}

                self.network[l]['delta'] = np.multiply(derivative,
                    np.dot(post_delta, post_weights.T)) # batch_size * n_unit^{l}


    def _update_weight(self, X, y):

        """
        w^{l}:  n_unit^{l-1} * n_unit^{l}
        b^{l}:  1 * n_unit^{l}
        """

        batch_size = len(X)

        for l in range(self.n_hidden_layer + 1):
            layer = self.network[l]

            # hidden layer
            if l == 0:
                pre_layer_output = X  # batch_size * n_unit^{l-1}
            else:
                pre_layer_output = self.network[l - 1]['output']  # batch_size * n_unit^{l-1}

            delta = layer['delta']  # batch_size * n_unit^{l}

            delta_weights = 1. / batch_size * \
                np.dot(pre_layer_output.T, delta) # n_unit^{l-1} * n_unit^{l}

            self.network[l]['weights'] = self.network[l]['weights'] \
                - self.learning_rate * delta_weights # n_unit^{l-1} * n_unit^{l}

            delta_bias = np.sum(delta, axis=0) / batch_size # 1 * n_unit^{l}
            self.network[l]['bias'] = self.network[l]['bias'] \
                - self.learning_rate * delta_bias # 1 * n_unit^{l}


    def _derivative(self, x):
        # derivative of sigmoid function
        #print('x: %s\n' % x)
        return np.multiply(x, 1-x)


    def _activate(self, x):
        return 1. / (1. + np.exp(-x))


    def predict(self, X):

        """
        X: batch_size * n_input_unit

        return: y: batch_size * n_output_unit
        """

        pre_layer_output = X # batch_size * n_input_unit

        for layer in self.network:

            layer_weights = layer['weights'] # n_unit^{l-1} * n_unit^{l}
            layer_bias = layer['bias']  # 1 * n_unit^{l}

            # batch_size * n_unit^{l}
            layer_input = self._get_weighted_sum(pre_layer_output, layer_weights, layer_bias)
            layer_output = self._activate(layer_input) # batch_size * n_unit^{l}

            pre_layer_output = layer_output

        return layer_output


def main():

    # f(x) = 1 * x1 + 2 * x2 + 0.5

    X = [[1, 1], [-1, -1], [1, 0.5], [-2, 1], [-3, 1]]
    y = [[1], [0], [1], [1], [0]]

    X = np.array(X)
    y = np.array(y)

    reg = MLP()
    reg.fit(X, y)
    reg.summary()
    print(reg.predict(X))


if __name__ == '__main__':
    main()