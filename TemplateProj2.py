# Template program.
# Input: datafile, k (#clusters), r (#iterations)
# Each row in datafile is a point. k is an integer. r is an integer.
# Output: cluster number of each data points

import sys
import numpy as np
from numpy import zeros, random

if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output_file')
    sys.exit()

#Read inputs.

# one row one data point, columns are dimentions
dataSet = sys.argv[1]

# separate or clustering data int to k groups, clasters.
# what ever you call it
k = sys.argv[2]

# init centroids with random samples  
#在样本集中随机选取k个样本点作为初始质心
def initCentroids(dataSet, k):  
	numSamples, dim = dataSet.shape   #矩阵的行数、列数 
	centroids = zeros((k, dim))  		#感觉要不要你都可以
	for i in range(k):  
		index = int(random.uniform(0, numSamples))  #随机产生一个浮点数，然后将其转化为int型
		centroids[i, :] = dataSet[index, :]  
	return centroids 




# r iterations: run the algorithm r times.
# at each time the algorithm should converge
r = sys.argv[3]

#Please finish the algorithm

#Save output in a comma separated file. 
#File name should pass from command line.

labels = np.array([1, 2, 3])
np.savetxt(sys.argv[4], labels, delimiter=',')
