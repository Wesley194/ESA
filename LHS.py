import numpy as np
from numpy import random
def LHS(sample:int,dimension:int,lowerbound,upperbound):
    x = np.empty([sample,dimension])
    slide = (upperbound-lowerbound)/sample
    for i in range(sample) : x[i] = random.uniform(lowerbound+i*slide, lowerbound+(i+1)*slide, size=dimension)
    row_indices_shuffled = np.argsort(np.random.rand(sample, dimension), axis=0)
    col_indices = np.arange(dimension)
    x = x[row_indices_shuffled, col_indices]
    return x
'''test:
x=LHS(10,5,0,10)
print(x)'''
