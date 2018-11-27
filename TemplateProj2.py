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

#Please finish the algorithm

#Save output in a comma separated file. 
#File name should pass from command line.
labels = np.array([1, 2, 3])
np.savetxt('TemplateProj2.txt', labels, delimiter=',')
