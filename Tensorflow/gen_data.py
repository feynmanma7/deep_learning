#encoding:utf-8
import sys

gen_number = 1000
if len(sys.argv) > 1: 
	gen_number = int(sys.argv[1])	

import numpy as np

w1 = 0.3
w2 = 0.7
w3 = 0.4
w4 = 0.2
b = 0.2

W = np.array([w1, w2, w3, w4]).reshape((4, 1))

for i in range(gen_number):
	noise = np.random.normal(0, 0.001)

	X = np.random.random(4)
	y = np.sum(np.multiply(W, X)) + b + noise

	print('%s,%s' % (
		','.join(list(map(lambda f:str(f), X))), y))
