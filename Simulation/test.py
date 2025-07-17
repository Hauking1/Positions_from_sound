import numpy as np

a = np.array([1,2,3,4,5,6])
print(np.reshape(a,shape=(3,2)))
print(np.reshape(a,shape=(3,2),order='F'))