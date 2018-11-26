import numpy as np

# Read numpy array from a file

# The file f1.txt has two rows. First row is 1  2 seond row is 3 4
X = np.genfromtxt('iris.data') # default delimiter is space
print("X=", X)

# The file f2.txt : First row is 1, 2 seond row is 3, 4
X = np.genfromtxt('iris.data', delimiter=',', autostrip=True) # strip spaces
print("X=", X)

# Write array to a file
np.savetxt('g1.txt', X) # default delimiter is space
np.savetxt('g2.txt', X, delimiter=',')
