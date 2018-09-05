import numpy as np
x = np.array([1,1,1,2,2,2,5,25,1,1])
y = np.bincount(x)

print y

ii = np.nonzero(y)[0]

print zip(ii,y[ii])
