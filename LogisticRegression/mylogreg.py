import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel, plot
from scipy.optimize import fmin_bfgs


# DEFINE SOME FUNCTIONS ##########################

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
def cost(theta, X, y, m):

    [lh, lhc] = logsig(theta, X, y)
    J = ((np.dot(-y.T, lh)) - (np.dot((1.0 - y.T), lhc)))
    J = J.sum() / m

    return J

# This function computes the gradient (grad) of the cost function
def comp_grad(theta, X, y, m):

    h = sigmoid(theta, X, m)
    grad = np.dot((h.T-y), X) / m

    return grad

# This function is called by the optimization package and provides J and grad
def f(theta):
    return cost(theta, it, y, m)

def fprime(theta):
    return comp_grad(theta, it, y, m)

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
data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

# Plot
pos = np.where(y == 1)
neg = np.where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='x', c='k')
scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Admitted', 'Not Admitted'])
#show()

m, n = X.shape
# m : number of training examples (in this case, =100),
#  n : number of features (in this case, =2)

# Add intercept term to x
it = np.ones(shape=(m, n+1))
it[:, 1:n+1] = X

# Initialize theta parameters
theta = np.zeros(shape=(1, 3))

# call fmin_bfgs function to the minimization
theta = fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)
print("The values of the parameters that minimize the cost J are", theta)

# Plot the decision boundary
plot_x = np.array([min(it[:, 1]) - 2, max(it[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plot(plot_x, plot_y)
legend(['Decision Boundary', 'Not admitted', 'Admitted'])
show()

# Example: computing the probability of admission of a student with
# scores 45 and 84
prob = sigmoid(np.array(theta), np.array([1.0, 45.0, 85.0]), 1)
print("For example, for a student with scores 45 and 85, we predict an admission probability of", prob)

# Compute classification accuracy for the training set
p = predict(np.array(theta), it)
A = pred_accuracy(p, y, m)
print("The classification accuracy on the training set is", A)




