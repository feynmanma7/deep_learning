#encoding:utf-8
import numpy as np

def softmax(x):

    exp_x = np.exp(x)
    exp_sum = np.sum(exp_x)
    return np.divide(exp_x, exp_sum)


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4])
    print(softmax(x))