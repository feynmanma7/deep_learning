#encoding:utf-8
from activation import sigmoid

class Layer:

    def __init__(self,
                 activation,
                 initializer):
        self.activation = activation
        self.initializer = initializer


class Dense(Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_data,
                 learning_rate=1e-3,
                 activation=None,
                 initializer=None,
                 ):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_data = input_data
        self.learning_rate = learning_rate
        self.activation = activation
        self.initializer = initializer

        # initialize
        self.weights = np.random.random(input_dim, output_dim)
        self.bias = np.random.random()


    def _update(self, loss):

        '''
        f_in^{L}(j) = (\sum_{i=1}^{n_in} w_{ij} * x_i) + b

        f_out^{L}(j) = activation(f_in^{L}(j)), take sigmoid as example,

        For regression, loss = 1/2 * (y_j - f_out(j)) ** 2

        ### for w_{ij}
        delta(w_{ij}) =
            \partial(loss) / \partial(w_{ij}) =
            \partial(loss)     / \partial(f_out(j))
          * \partial(f_out(j)) / \partial(f_in(j))
          * \partial(f_in(j))  / \partial(w_{ij})

        =   (f_out(j) - y_j)
          * (sigmoid(f_in(j)) * (1 - sigmoid(f_in(j)))
          * x_i

        ### for w_0
        delta_b =
            \partial(loss) / \partial(b) =
            \partial(loss)     / \partial(f_out(j))
          * \partial(f_out(j)) / \partial(f_in(j))
          * \partial(f_in(j))  / b)

        =   (f_out(j) - y_j)
          * (sigmoid(f_in(j)) * (1 - sigmoid(f_in(j)))
          * 1


        w_{ij} -= learning_rate * delta(w_{ij})
        b -= learning_rate * delta(b)


        '''


    def predict(self, X):
        preds = []
        for x in X:
            pred = sigmoid(np.dot(self.weights, x) + self.bias)
            preds.append(pred)

        return preds



