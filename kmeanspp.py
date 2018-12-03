
import sys
from numpy import *  
import time  
import matplotlib.pyplot as plt

if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output_file')
    sys.exit()

#Read inputs.

# one row one data point, columns are dimentions
dataSet = genfromtxt(sys.argv[1], delimiter=',', autostrip=True) # strip spaces

# separate or clustering data int to k groups, clusters.
k = int(sys.argv[2])

# r iterations: run the algorithm r times.
# at each time the algorithm should converge
r = int(sys.argv[3])

# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
	return sqrt(sum(power(vector2 - vector1, 2)))

DEBUG = False

def dist(v1, v2):
    return linalg.norm(v1 - v2)

def kmeanspp(X, k, W=None):
    if W == None:
        W = [1] * len(X)
    centroids_indices = [random.choice(len(X), 1, p=[w / sum(W) for w in W])[0]]
    
    for i in range(k - 1):
        
        x_dist_to_closest_centroids = [W[i] * min(dist(X[i], X[c]) ** 2 for c in centroids_indices) for i in
                                        range(len(X))]
        x_probs = [d / sum(x_dist_to_closest_centroids) for d in x_dist_to_closest_centroids]
        
        try:
            centroids_indices.append(random.choice(len(X), 1, p=x_probs)[0])
        except:
            print(x_probs)
            
    numSamples, dim = X.shape
    res = zeros((k, dim))

    for i in range(k):
        res[i, :] = X[centroids_indices[i], :]
    
    return res

def kmeansIter(dataSet, k, r):
	
	num = 1
	while num <= r:
		numSamples = dataSet.shape[0]
		# first column stores which cluster this sample belongs to,  
		# second column stores the error between this sample and its centroid  
		clusterAssment = mat(zeros((numSamples, 2)))
		clusterChanged = True  
	
		## step 1: init centroids  
		centroids = kmeanspp(dataSet, k, W=None)
		
		while clusterChanged:  
			clusterChanged = False  
			## for each sample  
			for i in range(numSamples):
				minDist  = float("inf")
				minIndex = 0  
				## for each centroid  
				## step 2: find the centroid who is closest  
				for j in range(k):  
					distance = euclDistance(centroids[j, :], dataSet[i, :])  
					if distance < minDist:  
						minDist  = distance  
						minIndex = j  
				
				## step 3: update its cluster
				if clusterAssment[i, 0] != minIndex:  
					clusterChanged = True  
					clusterAssment[i, :] = minIndex, minDist**2
			
	
			## step 4: update centroids  
			for j in range(k):  
				#clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
				pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]] #将dataSet矩阵中相对应的样本提取出来 
				centroids[j, :] = mean(pointsInCluster, axis = 0)  #计算标注为j的所有样本的平均值
			
		print(f'Iteration {num} complete!')  
		 
		err = 0
		for i in list(clusterAssment[:, 1]):
			err += float(i)		
		print(err) 
		
		num += 1

	return clusterAssment[:, 0]

#Save output in a comma separated file. 
#File name should pass from command line.
res = kmeansIter(dataSet, k, r)

savetxt(sys.argv[4], res, delimiter=',')