# Linear equations, eigenvectors, svd
import numpy as np

A = np.matrix('1 2; 3 4'); print("A=", A)
b = np.array([1,2]); print("b=", b)
# solve Ax=b
x = np.linalg.solve(A,b); print("x=",x)

# matrix inverse and pseudo inverse
A_inv = np.linalg.inv(A)
A_pinv = np.linalg.pinv(A)
print("A_inv=", A_inv, "\nA_pinv=",A_pinv)

# eigensystem of a symmetric matrix
B = np.matrix('1 0 0; 0 3 0; 0 0 2') 
print("B=", B)
evals,evecs = np.linalg.eigh(B) 
print("evals=", evals, " evecs=", evecs)
# eigenvalues in increasing order, not decreasing order. Sort them.

idx = np.argsort(evals)[::-1] # sort in reverse order
evals = evals[idx]
evecs = evecs[:,idx]
print("evals=", evals, " evecs=", evecs) # evectors are the cols of evecs

# extract the 2 dominant eigenvectors
r = 2
V_r = evecs[:,:r]; print("V_r=",V_r) # get first r eigenvectors


# eigensystem of a not necessarily symmetric matrix
B = np.matrix('1 0 0; 0 3 0; 0 0 2') 
A = np.matrix('1 1 1; 1 1 1; 1 1 1') 
C = np.dot(np.linalg.pinv(B), A)
print("C=", C)

# evals,evecs = np.linalg.eigh(C); this is wrong because C is not symmetric
evals,evecs = np.linalg.eig(C) # eigenvalues may be complex. Assume real
evals,evecs = np.linalg.eig(C)
print("evals=", evals, " evecs=", evecs)
# eigenvalues may be complex. Assume real, and sort them.

idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:,idx]
print("evals=", evals, " evecs=", evecs) # evectors are the cols of evecs


# computing SVD
# C = U S Vt
U,s,Vt = np.linalg.svd(C)
print("U=", U, "\n s=", s, "\n Vt=",Vt)





