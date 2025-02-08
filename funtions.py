import numpy as np
from scipy.integrate import quad
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1.1 Vector Addition
def vector_addition(u, v):
    return u + v

# 1.2 Scalar Multiplication of a Vector
def scalar_multiplication(alpha, v):
    return alpha * v

# 1.3 Dot Product
def dot_product(u, v):
    return np.dot(u, v)

# 1.4 Cross Product (3D)
def cross_product(u, v):
    return np.cross(u, v)

# 1.5 Norm of a Vector (Euclidean)
def vector_norm(v):
    return np.linalg.norm(v)

# 1.6 Orthogonality Condition
def is_orthogonal(u, v):
    return np.isclose(dot_product(u, v), 0)

# 1.7 Matrix Addition
def matrix_addition(A, B):
    return A + B

# 1.8 Matrix Scalar Multiplication
def matrix_scalar_multiplication(alpha, A):
    return alpha * A

# 1.9 Matrix-Vector Multiplication
def matrix_vector_multiplication(A, v):
    return np.dot(A, v)

# 1.10 Matrix Multiplication
def matrix_multiplication(A, B):
    return np.dot(A, B)

# 1.11 Transpose of a Matrix
def matrix_transpose(A):
    return A.T

# 1.12 Determinant of a 2x2 Matrix
def determinant_2x2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

# 1.13 Inverse of a 2x2 Matrix
def inverse_2x2(A):
    det = determinant_2x2(A)
    if det == 0:
        raise ValueError("Matrix is singular")
    return (1 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])

# 1.14 Kronecker Product
def kronecker_product(A, B):
    return np.kron(A, B)

# 2.1 Conditional Probability
def conditional_probability(P_A_and_B, P_B):
    return P_A_and_B / P_B

# 2.2 Law of Total Probability
def law_of_total_probability(P_A_given_Bi, P_Bi):
    return sum(P_A_given_Bi[i] * P_Bi[i] for i in range(len(P_Bi)))

# 2.3 Bayes' Theorem
def bayes_theorem(P_A_and_B, P_B):
    return P_A_and_B / P_B

# 2.4 Expectation
def expectation(X, P_X):
    return sum(x * p for x, p in zip(X, P_X))

# 2.5 Variance
def variance(X, P_X):
    mu = expectation(X, P_X)
    return sum((x - mu)**2 * p for x, p in zip(X, P_X))

# 2.6 Standard Deviation
def standard_deviation(X, P_X):
    return np.sqrt(variance(X, P_X))

# 2.9 Probability Mass Function (PMF)
def pmf(x, X, P_X):
    return P_X[X.index(x)] if x in X else 0

# 2.10 Probability Density Function (PDF)
from scipy.stats import norm
def pdf(x, mean=0, std=1):
    return norm.pdf(x, loc=mean, scale=std)

# 2.11 Joint Probability
def joint_probability(P_A_given_B, P_B):
    return P_A_given_B * P_B

# 2.12 Cumulative Distribution Function (CDF)
def cdf(x, mean=0, std=1):
    return norm.cdf(x, loc=mean, scale=std)

# 2.13 Entropy (Discrete)
def entropy(P_X):
    return -sum(p * np.log2(p) for p in P_X if p > 0)

# 2.19 Binary Cross-Entropy
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 3.1 Limit Definition of Derivative
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# 3.2 Power Rule
def power_rule_derivative(n):
    return lambda x: n * x**(n - 1)

# 3.5 Chain Rule
def chain_rule(f_prime, g, g_prime, x):
    return f_prime(g(x)) * g_prime(x)

# 3.6 Logarithmic Derivative
def log_derivative(x):
    return 1 / x

# 3.7 Exponential Derivative
def exp_derivative(x):
    return np.exp(x)

# 3.8 Integral of a Power Function
def integral_power_function(n, x):
    return x**(n + 1) / (n + 1)

# 3.19 Arc Length of a Curve
def arc_length(f_prime, a, b):
    integrand = lambda x: np.sqrt(1 + f_prime(x)**2)
    return quad(integrand, a, b)[0]

# 3.20 Curvature of a Function
def curvature(f_prime, f_double_prime, x):
    numerator = abs(f_double_prime(x))
    denominator = (1 + f_prime(x)**2)**1.5
    return numerator / denominator

# 4.1 Gradient Descent
def gradient_descent(gradient, theta, eta, steps):
    for _ in range(steps):
        theta -= eta * gradient(theta)
    return theta

# 4.2 Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(gradient, theta, eta, data, steps):
    for _ in range(steps):
        i = np.random.randint(len(data))
        theta -= eta * gradient(theta, data[i])
    return theta

# 4.3 Momentum-based Gradient Descent
def momentum_gradient_descent(gradient, theta, eta, beta, steps):
    v = 0
    for _ in range(steps):
        v = beta * v - eta * gradient(theta)
        theta += v
    return theta

# 5.1 Linear Regression Hypothesis
def linear_regression_hypothesis(X, beta):
    return X @ beta

# 5.2 Ordinary Least Squares (OLS)
def ols(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# 5.3 Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 5.14 Logistic Regression Hypothesis
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_hypothesis(X, beta):
    return sigmoid(X @ beta)

# 6.1 Perceptron Update Rule
def perceptron_update(w, x, y, y_pred, eta):
    return w + eta * (y - y_pred) * x

# 6.3 Sigmoid Activation
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

# 6.5 ReLU Activation
def relu_activation(z):
    return np.maximum(0, z)

# 6.9 Softmax Function
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / exp_z.sum(axis=0)

# 7.1 Euclidean Distance
def euclidean_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))

# 7.5 k-Means Objective
def k_means_objective(X, centroids, labels):
    return np.sum(np.linalg.norm(X - centroids[labels], axis=1)**2)

# 7.10 Silhouette Score
def silhouette_score_clustering(X, labels):
    return silhouette_score(X, labels)

# 8.1 Principal Component Analysis (PCA)
def pca(X, n_components):
    pca_model = PCA(n_components=n_components)
    return pca_model.fit_transform(X)

# 8.4 Singular Value Decomposition (SVD)
def svd(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt

# 10.2 Discounted Return
def discounted_return(rewards, gamma):
    G = 0
    for t, r in enumerate(rewards):
        G += (gamma**t) * r
    return G

# 10.8 Q-Learning Update
def q_learning_update(Q, state, action, reward, next_state, alpha, gamma):
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])