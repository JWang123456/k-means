

import sys
import numpy as np

if len(sys.argv) != 5 :
    print('usage : ', sys.argv[0], 'data_file labels_file reducedim2_vector_file reducedim2_reduced_data_file')
    sys.exit()


filename = sys.argv[1]

with open(filename,"r") as filestream:
	Xt = np.loadtxt(filestream, delimiter=',')

filename = sys.argv[2]

with open(filename,"r") as filestream:
	y = np.loadtxt(filestream, delimiter=',')


n,m = Xt.shape


unique_elements, counts_elements = np.unique(y, return_counts=True)



x = unique_elements.size
# calculating each group mean

mean = np.zeros((x,m))
#summ = np.zeros((1,m))
#count = 0
z = 0


for j in unique_elements:
    s = np.zeros((1,m))
    count = 0
    for i,xt in enumerate(Xt):
        if(y[i]==j):
            s +=xt
            count += 1
    mean[z] = s/count
    z += 1
        
      
# within class scatter initialisation
W = np.zeros((m,m));       

# calculating within each cluster scatter

#S = np.zeros((m,m))
z = 0
for j in unique_elements:
    S = np.zeros((m,m))
    for i,xt in enumerate(Xt):
        if(y[i]==j):
            st = xt - mean[z]
            stT = st.reshape(len(st),1)
            st = st.reshape(1,len(st))
            S += np.dot(stT, st)
    z += 1
    W += S



mixTureMean = Xt.mean(axis=0)

B = np.zeros((m,m))
j=0
for i,mt in enumerate(mean):
    z = (mt-mixTureMean)
    zT = z.reshape(len(z),1)
    z = z.reshape(1,len(z))
    B += np.dot(zT,z)*counts_elements[j]
    j +=1

W_ = np.linalg.inv(W)
#print(W)
#print(B)
#print(W_)
C = np.dot(W_,B)

#print(C)
U,S,V=np.linalg.svd(C,full_matrices=False)
V = U[:,0:2]
Vt = np.transpose(V);

X_Projection = np.dot(Vt,Xt.T);
D = np.transpose(X_Projection);

np.savetxt(sys.argv[4], D, delimiter=',')
np.savetxt(sys.argv[3], Vt, delimiter=',')

#save output in comma separated filename.txt. filename depends on the program
#np.savetxt('reduced-scatter3.txt', D, delimiter=',')
#np.savetxt('V-scatter3.txt', Vt, delimiter=',')



'''L,Q = np.linalg.eig(B)
L_sqrt = np.sqrt(L)
L_sqrt_inverse = np.reciprocal(L_sqrt)
L_sqrt_inverse = np.diag(L_sqrt_inverse)
C = L_sqrt_inverse@Q.T@W@Q@L_sqrt_inverse
sigma, U = np.linalg.eig(C)
v1 = Q@L_sqrt_inverse@U[:,0:1]
v2 = Q@L_sqrt_inverse@U[:,1:2]'''
