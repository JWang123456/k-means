
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

# separate or clustering data int to k groups, clasters.
# what ever you call it
k = int(sys.argv[2])

# r iterations: run the algorithm r times.
# at each time the algorithm should converge
r = int(sys.argv[3])

# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
	return sqrt(sum(power(vector2 - vector1, 2)))  #求这两个矩阵的距离，vector1、2均为矩阵

DEBUG = False

def dist(v1, v2):
    return linalg.norm(v1 - v2)


def kmeanspp(X, k, W=None):
    if W == None:
        W = [1] * len(X)
    centroids_indices = [random.choice(len(X), 1, p=[w / sum(W) for w in W])[0]]
    if DEBUG:
        print('First centroid index drawn at random:')
        print(centroids_indices)
    for i in range(k - 1):
        if DEBUG:
            print('iteration i = ' + str(i + 1))
        x_dist_to_closest_centroids = [W[i] * min(dist(X[i], X[c]) ** 2 for c in centroids_indices) for i in
                                        range(len(X))]
        x_probs = [d / sum(x_dist_to_closest_centroids) for d in x_dist_to_closest_centroids]
        if DEBUG:
            print('Distribution at iteration:')
            print(x_probs)
        try:
            centroids_indices.append(random.choice(len(X), 1, p=x_probs)[0])
            if DEBUG:
                print('centorids indices at the end of iteration')
                print(centroids_indices)
        except:
            print(x_probs)
            
    numSamples, dim = X.shape   #矩阵的行数、列数 
    res = zeros((k, dim))

    for i in range(k):
        res[i, :] = X[centroids_indices[i], :]
    
    return res

# def initCentroids(dataSet, k, centroidsIndex):  
# 	numSamples, dim = dataSet.shape   #矩阵的行数、列数 
# 	centroids = zeros((k, dim))  		#感觉要不要你都可以
# 	for i in centroidsIndex:  
# 		centroids[i, :] = dataSet[i, :]  
# 	return centroids



def kmeansIter(dataSet, k, r):
	
	num = 1
	while num <= r:
		numSamples = dataSet.shape[0]  #读取矩阵dataSet的第一维度的长度,即获得有多少个样本数据
		# first column stores which cluster this sample belongs to,  
		# second column stores the error between this sample and its centroid  
		clusterAssment = mat(zeros((numSamples, 2)))  #得到一个N*2的零矩阵
		clusterChanged = True  
	
		## step 1: init centroids  
		centroids = kmeanspp(dataSet, k, W=None)  #在样本集中随机选取k个样本点作为初始质心
		# print('centroids', centroids)
		
		while clusterChanged:  
			clusterChanged = False  
			## for each sample  
			for i in range(numSamples):  #range
				minDist  = 100000.0  
				minIndex = 0  
				## for each centroid  
				## step 2: find the centroid who is closest  
				#计算每个样本点与质点之间的距离，将其归内到距离最小的那一簇
				for j in range(k):  
					distance = euclDistance(centroids[j, :], dataSet[i, :])  
					if distance < minDist:  
						minDist  = distance  
						minIndex = j  
				
				## step 3: update its cluster 
				#k个簇里面与第i个样本距离最小的的标号和距离保存在clusterAssment中
				#若所有的样本不在变化，则退出while循环
				if clusterAssment[i, 0] != minIndex:  
					clusterChanged = True  
					clusterAssment[i, :] = minIndex, minDist**2  #两个**表示的是minDist的平方
			
	
			## step 4: update centroids  
			for j in range(k):  
				#clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
				
				pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]] #将dataSet矩阵中相对应的样本提取出来 
				
				centroids[j, :] = mean(pointsInCluster, axis = 0)  #计算标注为j的所有样本的平均值
			
		print(f'Iteration {num} complete!')  
		# return centroids, clusterAssment 
		err = 0
		
		for i in clusterAssment[:, 1]:
			err += i		
		print(err) 
		
		num += 1
		
	return reshape(clusterAssment[:, 0], (1, numSamples)).astype(int)

# print(res)
#Save output in a comma separated file. 
#File name should pass from command line.

res = kmeansIter(dataSet, k, r)
# labels = array([1, 2, 3])
savetxt(sys.argv[4], res, delimiter=',')


# show your cluster only available with 2-D data 
#centroids为k个类别，其中保存着每个类别的质心
#clusterAssment为样本的标记，第一列为此样本的类别号，第二列为到此类别质心的距离 
def showCluster(dataSet, k, centroids, clusterAssment):  
	numSamples, dim = dataSet.shape  
	if dim != 2:  
		print ("Sorry! I can not draw because the dimension of your data is not 2!")  
		return 1  
  
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
	if k > len(mark):  
		print ("Sorry! Your k is too large! please contact wojiushimogui")  
		return 1 
     
  
    # draw all samples  
	for i in range(numSamples):  
		markIndex = int(clusterAssment[i, 0])  #为样本指定颜色
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
  
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids  
	for i in range(k):  
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
  
	plt.show() 