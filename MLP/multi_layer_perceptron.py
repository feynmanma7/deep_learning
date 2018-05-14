#encoding:utf-8
import numpy as np

"""
Initialization, init weights and bias of each layer_i to layer_{i+1}

Forward-propagation, predict output of each layer for the input.

Back-propagation, compute gradient of each weight.

Update weights

"""


class MLP:

    def __init__(self,
                 n_layers=3,
                 input_unit=4,
                 n_hidden_units=[3, 4, 3],
                 n_epochs=5,
                 batch_size=1,
                 learning_rate=1e-3):
        """
        Specified Networks.

        n_layers: number of hidder layers
        """

        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.network = []

        for l in range(n_layers):
            # L hidden layers and 1 output layer, layer_{-1} is the input layer.
            # Weights of layer_{l} link layer_{l-1} and layer_{l}

            layer = {}

            if l == 0:
                layer['weights'] = np.random.random((input_unit, n_hidden_units[l]))
            else:
                layer['weights'] = np.random.random((n_hidden_units[l-1], n_hidden_units[l]))

            layer['bias'] = np.random.random()
            self.network.append(layer)

        # ouput-layer, one unit for regression
        layer = {}
        layer['weights'] = np.random.random((n_hidden_units[-1], 1))
        layer['bias'] = np.random.random()
        self.network.append(layer)


    def summary(self):
        for i in range(self.n_layers):
            layer = self.network[i]
            print('Layer %s, weights: %s, bias: %s' \
                  % (i, layer['weights'], layer['bias']))


    def fit(self, X, y):

        for epoch in range(self.n_epochs):

            for batch in range(int(len(X) / self.batch_size)):

                X_mini_batch = X[batch * self.batch_size : (batch+1) * self.batch_size]
                y_mini_batch = y[batch * self.batch_size : (batch+1) * self.batch_size]

                self._forward_propagation(X_mini_batch)
                self._back_propagation(X_mini_batch, y_mini_batch)
                self._update_weight(X_mini_batch, y_mini_batch)

            loss = self._compute_loss(X, y)
            print('Epoch %s, loss=%s' % (epoch, loss))


    def _compute_loss(self, X, y):

        pred = self.predict(X)
        loss = 1. / len(X) * 1 / 2 * np.sum(np.subtract(pred, y) ** 2)
        return loss



    def _forward_propagation(self, X):

        pre_layer_output = X

        for i in range(self.n_layers + 1):
            layer = self.network[i]

            layer_weights = layer['weights']
            layer_bias = layer['bias']

            layer_input = np.dot(pre_layer_output, layer_weights) + layer_bias
            layer_output = self._activate(layer_input)

            layer['output'] = layer_output
            pre_layer_output = layer_output


    def _back_propagation(self, X, y):

        """
        # layer_{l}, l = L (the output layer)
        error_j^{l} = 1 / 2 * (y_j - predict_j) ** 2

        # layer_{l}, l = 0, 1, 2, ..., L-1;
        error_i^{l} = 1. / len(X) * \sum_j * weights_{ij}^{l+1} * error_j^{l+1}

        """


        for l in reversed(range(self.n_layers+1)):

            layer = self.network[l]
            #bias = layer['bias']

            if l == self.n_layers:
                layer_output = layer['output']
                errors = 0.5 * np.subtract(y, layer_output) ** 2

            else:
                post_weights = self.network[l+1]['weights']
                post_errors = self.network[l+1]['errors']

                errors = np.dot(post_errors.T, post_weights.T).T
                errors = 1. / len(X) * errors

            layer['errors'] = errors



    def _update_weight(self, X, y):

        """
        l = 0, 1, 2, ..., L-1, L

        weights_{ij}^l += weights_{ij}^{l} + learning_rate
            * \partial_{weights_{ij}^l}(error_j^l)

        \partial_{weights_{ij}} (error_j^l) =
                \partial_{input{ij}^l} (error_j^l)
            *   \partial_{weights_{ij}^l} (input_{ij}^l)

        =  error_j^l * (1 - error_j^l)
         *  output_{i}^{l-1}


        bias_{i}^l += bias_{i}^l + learning_rate
            * \partial_{bias_{i}^l} (error_j^l)

        \partial_{bias_{i}^l} (error_j^l) = 1. / len(X) * \sum_{j} error_j^l * (1 - error_j^l)

        """

        for l in range(self.n_layers+1):
            layer = self.network[l]
            errors = layer['errors']

            if l == 0:
                layer_output = X
            else:
                layer_output = self.network[l-1]['output']

            delta_weights = np.dot(layer_output.T,
                                   np.multiply(errors, np.subtract(1, errors)).T)
            layer['weights'] = np.add(layer['weights'],
                                      self.learning_rate * delta_weights)

            delta_bias = 1. / len(X) * np.sum(np.multiply(errors, np.subtract(1, errors)))
            layer['bias'] = np.add(layer['bias'],
                                         self.learning_rate * delta_bias)


    def _activate(self, x):
        return 1. / (1. + np.exp(-x))


    def _predict(self, x):
        result = x

        for layer in self.network:
            weights = layer['weights']
            bias = layer['bias']

            result = np.dot(result, weights) + float(bias)
            result = self._activate(result)

        return result


    def predict(self, X):
        results = []
        for x in X:
            results.append(self._predict(x))
        return results


def main():

    X = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    y = [1, 2, 3, 4, 5]

    X = np.array(X)
    y = np.array(y)

    reg = MLP()
    reg.summary()
    reg.fit(X, y)
    print(reg.predict(X))


if __name__ == '__main__':
    main()