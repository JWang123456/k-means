# Template program.
# Input: datafile, k (#clusters), r (#iterations)
# Each row in datafile is a point. k is an integer. r is an integer.
# Output: cluster number of each data points

import sys
import numpy as np

if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output_file')
    sys.exit()

#Read inputs.

# one row one data point, columns are dimentions
DATA = sys.argv[1]

# separate or clustering data int to k groups, clasters.
# what ever you call it
k = sys.argv[2]

# r iterations: run the algorithm r times.
# at each time the algorithm should converge
# yeah fine
r = sys.argv[3]

#Please finish the algorithm

#Save output in a comma separated file. 
#File name should pass from command line.

labels = np.array([1, 2, 3])
np.savetxt(sys.argv[4], labels, delimiter=',')
