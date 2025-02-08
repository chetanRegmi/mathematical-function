from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated dataset
X = np.random.rand(100, 2)  # Features (e.g., age, income)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Labels (binary classification)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize parameters
beta = np.zeros(X_train.shape[1])

# Define gradient for logistic regression
def logistic_gradient(beta, X, y):
    predictions = logistic_regression_hypothesis(X, beta)
    return X.T @ (predictions - y) / len(y)

# Train using gradient descent
beta = gradient_descent(logistic_gradient, beta, eta=0.1, steps=1000, args=(X_train, y_train))

# Predictions
y_pred_prob = logistic_regression_hypothesis(X_test, beta)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluate
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")