# init centroids with random samples  
#在样本集中随机选取k个样本点作为初始质心

import sys
import numpy as np


k = 10
dataSet = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True) # strip spaces
# print("X=", X)

def initCentroids(dataSet, k):  
	numSamples, dim = dataSet.shape   #矩阵的行数、列数 
	centroids = np.zeros((k, dim))  		#感觉要不要你都可以
	for i in range(k):  
		index = int(np.random.uniform(0, numSamples))  #随机产生一个浮点数，然后将其转化为int型
		centroids[i, :] = dataSet[index, :]  
	return centroids 


print(initCentroids(dataSet, k))