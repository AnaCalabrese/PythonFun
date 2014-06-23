import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import show
from scipy.optimize import fmin_bfgs

## Define some functions
def sigmoid(theta, X, m):
    theta_t = np.transpose(theta)
    z = np.dot(X, theta_t)

    h = 1.0 / (1.0 + np.exp(-1.0 * z))

    return h

def predict(Theta1, Theta2, X):
    m = X[:, 0].shape
    a2_temp = sigmoid(Theta1, X, m)

    m, n = a2_temp.shape
    a2 = np.ones(shape=(m, n+1))
    a2[:, 1:n+1] = a2_temp
    a3 = sigmoid(Theta2, a2, m)

    p = np.argmax(a3, axis=1)+1

    return p

def pred_accuracy(p, y, m):

    hit = 0
    for q in range(0, m):
        if p[q] == y[q]:
            hit = hit + 1

    accu = hit / m

    return accu


#################################

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



# Useful parameters
num_labels = 10  # number of classes
m, n = X.shape

# Add intercept term to X (O(0)) : 1
it = np.ones(shape=(m, n+1))
it[:, 1:n+1] = X

# Using the learned weight matrices Theta1 and Theta2, predict the
# labels in the training set using feed-forward propagation
p = predict(Theta1, Theta2, it)
A = pred_accuracy(p, y, m)
print("The classification accuracy on the training set is", A)