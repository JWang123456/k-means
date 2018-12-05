# Template program.
# Input: datafile, k (#clusters), r (#iterations)
# Each row in datafile is a point. k is an integer. r is an integer.
# Output: cluster number of each data points

import sys
import numpy as np
import random
import operator
import math


if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output_file')
    sys.exit()

#Read inputs.

# one row one data point, columns are dimentions
dataSet = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True) # strip spaces

# separate or clustering data int to k groups, clusters.
k = int(sys.argv[2])

# r iterations: run the algorithm r times.
# at each time the algorithm should converge
r = int(sys.argv[3])

# Number of data points
n = len(dataSet)

# Fuzzy parameter
m = 2.00

def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat


def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(dataSet[i, :])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
	return abs(sum(np.power(list(map(lambda x, y: x - y, vector2, vector1)), 2)))

def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(dataSet[i, :])
        distances = [euclDistance(x, cluster_centers[j]) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    num = 0
    while num <= r:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)

        print(f'Iteration {num} complete!')  
        
        err = 0
        for i in range(n):
            err += np.sqrt(sum(np.power(cluster_centers[cluster_labels[i]] - dataSet[i], 2)))
        print(err) 
        
        num += 1
    return cluster_labels


labels = fuzzyCMeansClustering()

#Save output in a comma separated file. 
#File name should pass from command line.
np.savetxt(sys.argv[4], labels, delimiter=',')
