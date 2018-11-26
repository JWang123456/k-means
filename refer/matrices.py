# use Numpy arrays instead of the native Python arrays
import numpy as np

# Create a matrix
X = np.matrix('1 2; 3 4') # 2x2 matrix. semicolon separates rows
print("X=", X) 

y = np.matrix('1;2'); print("y=", y) # 2x1 matrix
yt = np.matrix('1 2'); print("yt=", yt) # 1x2 matrix


v = np.array([1,2]) # 2 vector
print("v=", v) 

# Matrix can also be created from a 1D array (vector) by reshaping it
w = np.array([1,2,3,4,5,6]); print("w=", w) 
W = w.reshape(3, 2); print("W=", W) 

# Equivalently:
W = np.array([1,2,3,4,5,6]).reshape(3, 2); print("W=", W) 

# addition, subtraction, scalar multiplication, constant matrices
A = (2*X + 10)/2; print("A=", A) 

Xt = X.T; print("Xt=", Xt) # matrix transpose
vt = v.T; print("vt=", vt) # vector transpose doe not change the vector

# vector to matrix
Vt = v[None,:]; print("Vt=", Vt) # vector converted into a 1 row matrix
V = v[:,None]; print("V=", V) # vector converted into a 1 column matrix


# dot product of two matrices:
B = np.dot(X,X); print("B=", B) 

# Equivalently:
B = X*X ; print("B=", B)
B = X@X ; print("B=", B)

# C = np.dot(X,yt) gives an error because X is 2x2, yt is 1x2
C = np.dot(X,y); print("C=", C) 
C = X*y; print("C=", C) 

# vectors

# dot product of matrix and vector
d1 = np.dot(X,v); print("d1=", d1) # matrix dot vector: a vector. d1=X v
d2 = np.dot(v,X); print("d2=", d2) # vector dot matrix: a vector. d2=v'X=(X'v)'

# dot product of vector and vector

m = np.dot(v,v); print("m=",m) # vector dot vector is a scalar
m1 = v*v; print("m1=",m1) # this multiplies coordinates. m1[i] = v[i]*v[i]
m = np.sum(v*v); print("m=",m) # this also computes the dot product

# outer product of vector and vector

vvt = np.outer(v,v); print("vvt=",vvt)
vvt = np.dot(v,v.T); print("vvt=",vvt) # does not work because v.T is same as v
vvt = np.dot(v[:,None],v[None,:]); print("vvt=",vvt)

# matrix dimensions
m,n = vvt.shape; print("m=",m," n=",n)

# mean
mean_all = np.mean(X); print("mean_all=",mean_all) # scalar value of 2.5
mean_rows = np.mean(X,axis=0); print("mean_rows=",mean_rows) # 1x2 matrix
mean_cols = np.mean(X,axis=1); print("mean_cols=",mean_cols) # 2x1 matrix 

# Example: computing the covariance matrix of X
mu = np.mean(X, axis=1)
Xc = X - mu;  print("Xc=",Xc)
C = Xc*Xc.T;  print("C=",C)
