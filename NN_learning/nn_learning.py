# This code implements the backpropagation algorithm to learn the weights of a neuronal network

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import show
from scipy.optimize import fmin_cg

##### DEFINE SOME USEFUL FUNCTIONS #################################

def sigmoid(z):
    #theta_t = np.transpose(theta)
    #z = np.dot(X, theta_t)

    h = 1.0 / (1.0 + np.exp(-1.0 * z))

    return h

def sigmoid_grad(z):

    s_grad = sigmoid(z) * (1 - sigmoid(z))

    return s_grad

def cost(Theta, X, y, Lin, Lhidd, Lout, lam):


    Theta1 = np.reshape(Theta[0: (Lhidd * (Lin+1))], (Lhidd, Lin+1))
    Theta2 = np.reshape(Theta[(Lhidd * (Lin+1)): (Lhidd * (Lin+1))+(Lout * (Lhidd+1))], (Lout, Lhidd+1))


    # Forward computation
    m, n = X.shape
    a1 = np.ones(shape=(m, n+1))
    a1[:, 1:n+1] = X                  #  m x n+1 = 5000 x 401
    z2 = np.dot(a1, Theta1.T)         # (m x n+1) x (n+1 x h) = m x h = 5000 x 25
    m, h = z2.shape
    a2 = np.ones(shape=(m, h+1))
    a2[:, 1:h+1] = sigmoid(z2)        #  m x h+1 = 5000 x 26
    z3 = np.dot(a2, Theta2.T)         # (m x h+1) x (h+1 x k) = m x k = 5000 x 10
    a3 = sigmoid(z3)                  #  m x k
    m, k = a3.shape

    # Compute J using activations
    Y = np.zeros(shape=(m, k)) # m x k = 5000 x 10
    for i in range(0, m):
        Y[i, y[i]-1] = 1

    temp = np.sum((Y * np.log(a3)) + (1 - Y) * np.log(1-a3), axis=1)
    Jo = - np.sum(temp, axis=None) / m

    # Compute regularization term
    Theta1_2 = np.sum(np.sum((Theta1[:, 1:n+1] * Theta1[:, 1:n+1]), axis=1), axis=None)
    Theta2_2 = np.sum(np.sum((Theta2[:, 1:h+1] * Theta2[:, 1:h+1]), axis=1), axis=None)
    R = (lam / (2 * m)) * (Theta1_2 + Theta2_2)

    J = Jo + R

    return J

def comp_grad(Theta, X, y, Lin, Lhidd, Lout, lam):

    Theta1 = np.reshape(Theta[0: (Lhidd * (Lin+1))], (Lhidd, Lin+1))
    Theta2 = np.reshape(Theta[(Lhidd * (Lin+1)): (Lhidd * (Lin+1))+(Lout * (Lhidd+1))], (Lout, Lhidd+1))

    # Forward computation
    m, n = X.shape
    a1 = np.ones(shape=(m, n+1))
    a1[:, 1:n+1] = X                  #  m x n+1 = 5000 x 401
    z2 = np.dot(a1, Theta1.T)         # (m x n+1) x (n+1 x h) = m x h = 5000 x 25
    m, h = z2.shape
    a2 = np.ones(shape=(m, h+1))
    a2[:, 1:h+1] = sigmoid(z2)        #  m x h+1 = 5000 x 26
    z3 = np.dot(a2, Theta2.T)         # (m x h+1) x (h+1 x k) = m x k = 5000 x 10
    a3 = sigmoid(z3)                  #  m x k
    m, k = a3.shape

    Y = np.zeros(shape=(m, k)) # m x k = 5000 x 10
    for i in range(0, m):
        Y[i, y[i]-1] = 1

    # Backpropagation
    d3 = a3 - Y                                          # m x k = 5000 x 10
    d2 = np.dot(d3, Theta2[:, 1:h+1]) * sigmoid_grad(z2) # m x h = 5000 x 25

    D1 = np.dot(d2.T, a1) / m                            # h x n+1 = 25 x 401
    D2 = np.dot(d3.T, a2) / m                            # k x h+1 = 10 x 26

    # Compute regularization term
    R1 = lam * Theta1[:, 1:n+1] / m
    R2 = lam * Theta2[:, 1:h+1] / m

    # Add regularization
    D1[:, 1:n+1] = D1[:, 1:n+1] + R1
    D2[:, 1:h+1] = D2[:, 1:h+1] + R2

    # Unroll gradient
    grad = np.append(D1.ravel(),D2.ravel(), axis=0) # N = 10285

    return grad

def rand_init_weights(Lin, Lout):
    W = np.zeros((Lout, 1 + Lin))
    epsilon_init = 0.12
    W = np.random.rand(Lout, 1 + Lin) * 2 * epsilon_init - epsilon_init

    return W

def f(Theta):
    return cost(Theta, X, y, Lin, Lhidd, Lout, lam)

def fprime(Theta):
    return comp_grad(Theta, X, y, Lin, Lhidd, Lout, lam)

def predict(Theta1, Theta2, x):

    m,n = x.shape

    X = np.ones(shape=(m, n+1))
    X[:, 1:n+1] = x

    a2_temp = sigmoid(np.dot(X, Theta1.T))

    m, n = a2_temp.shape
    a2 = np.ones(shape=(m, n+1))
    a2[:, 1:n+1] = a2_temp
    a3 = sigmoid(np.dot(a2, Theta2.T))

    p = np.argmax(a3, axis=1)+1

    return p

def pred_accuracy(p, y, m):

    hit = 0
    for q in range(0, m):
        if p[q] == y[q]:
            hit = hit + 1

    accu = hit / m

    return accu
####################################################################


# Load data
COLUMN_SEPARATOR = ','
X = pd.io.parsers.read_csv('X.csv', sep=COLUMN_SEPARATOR, header=None)
y = pd.io.parsers.read_csv('y.csv', sep=COLUMN_SEPARATOR, header=None)
Theta1 = pd.io.parsers.read_csv('Theta1.csv', sep=COLUMN_SEPARATOR, header=None)
Theta2 = pd.io.parsers.read_csv('Theta2.csv', sep=COLUMN_SEPARATOR, header=None)

# Convert X and y to arrays and eliminate dummy dimension of y
X = np.array(X) # (5000, 400) : 5000 20x20 images of digits 0:9
y = np.array(y) # (5000, 1) : labels (1:10)
y = y[:, 0]     # now (5000,)

Theta1 = np.array(Theta1) # (25, 401)
Theta2 = np.array(Theta2) # (10, 26)
# print("Theta 1 :", Theta1.shape)
# print("Theta 2 :", Theta2.shape)

# randomly select 100/5000 images and plot them
RP = np.random.permutation(5000)
R = RP[0:100]

j = 0
B = np.reshape(X[R[j], :], (20, 20))
B = B.T
for i in R[j+1:j+10]:
    b = np.reshape(X[i, :], (20, 20))
    B = np.concatenate((B, b.T), axis=1)
C = B

B = np.reshape(X[R[j], :], (20, 20))
B = B.T
for j in range(10, 100, 10):

    B = np.reshape(X[R[j], :], (20, 20))
    B = B.T
    for i in R[j+1:j+10]:
        b = np.reshape(X[i, :], (20, 20))
        B = np.concatenate((B, b.T), axis=1)

    C = np.vstack((C, B))
imgplot = plt.imshow(C)
show()

######

# Some useful parameters
Lin   = 400
Lhidd = 25
Lout  = 10
m, n  = X.shape

# Define penalty
lam = 1

# Using the weights Theta1 and Theta2, compute the cost function using feedforward propagation
# Unroll Thetha1 and Theta2
Theta = np.append(Theta1.ravel(), Theta2.ravel(), axis=0) # N = 10285
# print("Unrolled Theta :", Theta.shape)

J = cost(Theta, X, y, Lin, Lhidd, Lout, lam)
print("Cost function with lamda =", lam, "is", J)

grad = comp_grad(Theta, X, y, Lin, Lhidd, Lout, lam)
# print("Unrolled Gradient :", grad.shape)

# Initialize Theta1 and Theta2
Theta1_init = rand_init_weights(Lin, Lhidd)
Theta2_init = rand_init_weights(Lhidd, Lout)
# print("init theta 1 :", Theta1_init.shape)
# print("init theta 2 :", Theta2_init.shape)

Theta_init = np.append(Theta1_init.ravel(), Theta2_init.ravel(), axis=0) # N = 10285
# print("init Theta unrolled :", Theta_init.shape)

theta_out = fmin_cg(f, Theta_init, fprime, disp=True, maxiter=50)
# print("Optimum theta", theta_out.shape)

theta1_out = np.reshape(theta_out[0: (Lhidd * (Lin+1))], (Lhidd, Lin+1))
theta2_out = np.reshape(theta_out[(Lhidd * (Lin+1)): (Lhidd * (Lin+1))+(Lout * (Lhidd+1))], (Lout, Lhidd+1))

p = predict(theta1_out, theta2_out, X)
A = pred_accuracy(p, y, m)
print("The classification accuracy on the training set is", A)