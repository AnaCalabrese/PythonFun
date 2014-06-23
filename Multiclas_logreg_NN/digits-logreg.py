import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import show
from scipy.optimize import fmin_bfgs

## DEFINE SOME FUNCTIONS ####

# This function computes the sigmoid hypothesis (h) for logistic regression
def sigmoid(theta, X, m):
    theta_t = np.transpose(theta)
    z = np.dot(X, theta_t)

    h = 1.0 / (1.0 + np.exp(-1.0 * z))

    return h


# This function computes the log(h) and also log(1-h) used for computing the cost
def logsig(theta, X, y):

    theta_t = np.transpose(theta)
    z = np.dot(X, theta_t)

    logh = -np.log(1.0 + np.exp(-1.0 * z))
    loghc = (-1.0 * z) - np.log(1.0 + np.exp(-1.0 * z))

    return logh, loghc

# This function computes the cost function (J) for logistic regression
def cost(theta, X, y, m, n, lam):

    [lh, lhc] = logsig(theta, X, y)
    J = ((np.dot(-y.T, lh)) - (np.dot((1.0 - y.T), lhc)))
    J = J.sum() / m
    Th = theta[1:n] ** 2
    R = Th.sum() * (lam / (2*m))

    J = J + R

    return J

# This function computes the gradient (grad) of the cost function
def comp_grad(theta, X, y, m, n, lam):

    h = sigmoid(theta, X, m)
    grad = np.dot((h.T-y), X)

    R = np.zeros(shape=(1, n))
    R[0, 1:n] = lam * theta[1:n].T

    G = (grad + R) / m

    return G[0]


# This function computes the probability of admission for the training set using the learned logistic
# regression parameters
def predict(theta, X):

    m, n = X.shape
    h = sigmoid(all_theta, it, m)
    p = np.argmax(h, axis=1)+1

    return p

# This function computes the classification performance for the training set
def pred_accuracy(p, y, m):

    hit = 0
    for q in range(0, m):
        if p[q] == y[q]:
            hit = hit + 1

    accu = hit / m

    return accu

##################################################

# Load data and plot
COLUMN_SEPARATOR = ','
X = pd.io.parsers.read_csv('X.csv', sep=COLUMN_SEPARATOR, header=None)
y = pd.io.parsers.read_csv('y.csv', sep=COLUMN_SEPARATOR, header=None)

# Convert X and y to arrays
X = np.array(X) # (5000, 400) : 5000 20x20 images of digits 0:9
y = np.array(y) # (5000, 1) : labels (1:10)

y = y[:, 0] # reshape y a little


# plot an example image
r = 4000
a = X[r, :]
b = np.reshape(a, (20,20))
#imgplot = plt.imshow(b.T)
#show()

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

#######################################################################
## Classify one vs all

# Useful parameters
num_labels = 10  # number of classes
m, n = X.shape

# Add intercept term to X (O(0)) : 1
it = np.ones(shape=(m, n+1))
it[:, 1:n+1] = X

m, n = it.shape

# Initialize theta parameters
theta = np.zeros(shape=(1, n))

# Choose penalty
lam = 0.1

# call fmin_bfgs function to the minimization
all_theta = np.zeros(shape=(num_labels, n))
for label in range(1, num_labels+1):

    # This function is called by the optimization package and provides J and grad
    def f(theta):

        bb = np.where(y == label)
        yb = np.zeros(shape=(m, ))
        yb[bb] = 1

        return cost(theta, it, yb, m, n, lam)

    def fprime(theta):

        bb = np.where(y == label)
        yb = np.zeros(shape=(m, ))
        yb[bb] = 1

        return comp_grad(theta, it, yb, m, n, lam)

    print("training class ... ", label)
    all_theta[label-1, :] = fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)

# Predict labels for the training set
p = predict(all_theta, it)
A = pred_accuracy(p, y, m)
print("The classification accuracy on the training set is", A)
