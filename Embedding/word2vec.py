# encoding:utf-8
import numpy as np

np.random.seed(20170430)

'''
CBOW: Continuous bag of words
'''

def indicator(a, b):
    return a == b if 1 else 0

class CBOW_one_word_context():
    n_V = 5
    n_h = 3
    eta = 1e-2 # learning rate
    n_epoch = 5

    def __init__(self):
        self.W = np.random.random((self.n_V, self.n_h))
        self.h = np.zeros((self.n_h))
        self.W_1 = np.random.random((self.n_h, self.n_V))
        self.y = np.zeros((self.n_V))

        self.exp_table = np.zeros((self.n_V))

    def _forward_propagation(self, in_idx, out_idx):
        sum = 0
        for i in range(self.n_V):
            self.exp_table[i] = np.exp(
                np.dot(self.W_1[:, i].reshape(1, self.n_h),
                       self.W[in_idx, :].reshape(self.n_h, 1)))
            sum += self.exp_table[i]

        self.h = self.W[in_idx]

        for i in range(self.n_V):
            self.y[i] = self.exp_table[i] / sum

    def _back_propagation(self, in_idx, out_idx):
        errors = np.zeros((self.n_V))
        for j in range(self.n_V):
            errors[j] = self.y[j] - indicator(j, out_idx)

        # EH
        sum = 0
        for j in range(self.n_V):
            sum += errors[j] * self.W_1[:, j]

        # Update W_1 [n_h, n_V]
        for j in range(self.n_V):
            self.W_1[:, j] = self.W_1[:, j] - \
                self.eta * errors[j] * self.h

        # Update W [n_V, n_h]
        self.W[in_idx, :] = self.W[in_idx, :] - \
                self.eta * sum

    def _one_epoch(self):
        for items in get_data():
            for in_idx, out_idx in get_input_output(items):

                self._forward_propagation(in_idx, out_idx)
                self._back_propagation(in_idx, out_idx)

    def fit(self):
        for _ in range(self.n_epoch):
            self._one_epoch()

        print(self.W)
        print(self.W_1)

    def find_similarity(self, item_idx):
        # in W_1 [n_h, n_V]
        item_vec = self.W_1[:, item_idx]

        sim = np.argsort(-np.dot(
            item_vec, self.W_1))

        sim = np.delete(sim, np.where(sim == item_idx))
        print(sim)


def get_input_output(arr):
    for i in range(len(arr) - 1):
        in_idx = arr[i]
        out_idx = arr[i + 1]

        yield in_idx, out_idx


def get_data():
    data = np.array([[0, 1, 2, 3, 4], [0, 3, 4],
                     [2, 4], [1, 2, 3],
                     [2, 3, 4], [4, 3, 1]])

    return data


if __name__ == '__main__':
    cb = CBOW_one_word_context()
    cb.fit()
    cb.find_similarity(2)

