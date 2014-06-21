import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel, contour
from scipy.optimize import fmin_bfgs

## DEFINE SOME FUNCTIONS ####
def mapFeature(X, m, n):

    max_order = 6
    N = 1+2+3+4+5+6+7 # total number of features (up to O(6)) including the bias and linear terms

    # First, add intercept term to X (O(0)) : 1
    Xm = np.ones(shape=(m, N))
    Xm[:, 1:n+1] = X

    q = n
    for R in range(n, max_order+1):
        for i in range(0, R+1):
            q = q + 1
            v = Xm[:, 1] ** (R-i)
            u = Xm[:, 2] ** i

            Xm[:, q] = v * u

    return Xm

# This function computes the sigmoid hypothesis (h) for logistic regression
def sigmoid(theta, X, m):
    theta_t = np.transpose(theta)
    z = np.dot(X, theta_t)
    h = np.zeros(m)

    if m > 1:
        for q in range(0, m):
            if z[q] >= 0:
                h[q] = 1.0 / (1.0 + np.exp(-1.0 * z[q]))
            else:
                h[q] = np.exp(z[q]) / (1.0 + np.exp(z[q]))
    else:
        h = 1.0 / (1.0 + np.exp(-1.0 * z))
    return h


# This function computes the log(h) and also log(1-h) used for computing the cost
def logsig(theta, X, y):

    theta_t = np.transpose(theta)
    z = np.dot(X, theta_t)
    logh = np.zeros(len(y))  # log(h)
    loghc = np.zeros(len(y)) # log(1-h)

    for q in range(0, len(z)):
        if z[q] >= 0:
            logh[q] = -np.log(1.0 + np.exp(-1.0 * z[q]))
            loghc[q] = (-1.0 * z[q]) - np.log(1.0 + np.exp(-1.0 * z[q]))

        else:
            logh[q] = z[q] - np.log(1.0 + np.exp(z[q]))
            loghc[q] = - np.log(1.0 + np.exp(z[q]))
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

# This function is called by the optimization package and provides J and grad
def f(theta):
    return cost(theta, Xm, y, m, n, lam)

def fprime(theta):
    return comp_grad(theta, Xm, y, m, n, lam)

# This function computes the probability of admission for the training set using the learned logistic
# regression parameters
def predict(theta, X):

    m, n = X.shape
    p = np.zeros(shape=(m, 1))
    h = sigmoid(theta, X, m)

    for q in range(0, m):
        if h[q] > 0.5:
            p[q, 0] = 1
        else:
            p[q, 0] = 0

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

# Load the dataset
data = np.loadtxt('ex2data2.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

# Plot
pos = np.where(y == 1)
neg = np.where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='x', c='k')
scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
xlabel('Microchip test 1')
ylabel('Microchip test 2')
legend(['y=1', 'y=0'])
#show()

# Add intercept term and features of up to O(6)
m, n = X.shape
Xm = mapFeature(X, m, n)

m, n = Xm.shape

# Initialize theta parameters
theta = np.zeros(shape=(1, n))

# choose penalization
lam = 1

#J = cost(theta, Xm, y, m, n, lam)
#print("The cost function with lamda = 1 and theta = 0 is J = ", J)

#G = comp_grad(theta, Xm, y, m, n, lam)
#print("The gradient of the cost function with lamda = 1 and theta = 0 is G = ", G)

# call fmin_bfgs function to the minimization
theta = fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)
print("The values of the parameters that minimize the cost J are", theta)

# Compute classification accuracy for the training set
p = predict(np.array(theta), Xm)
A = pred_accuracy(p, y, m)
print("The classification accuracy on the training set is", A)

# Plot Boundary
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
#U = np.array([u, v]).T
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        U = np.array([u[i], v[j]]).T
        z[i, j] = (mapFeature(U, 1, 2).dot(np.array(theta)))

z = z.T
contour(u, v, z)
#title('lambda = %f' % l)
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend(['y = 1', 'y = 0', 'Decision boundary'])
show()