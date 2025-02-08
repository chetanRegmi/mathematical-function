# Define the cost function
def cost_function(x):
    return x**2 + 3*x + 2

# Compute the derivative
def cost_function_derivative(x):
    return 2*x + 3

# Optimize using gradient descent
initial_x = 0
learning_rate = 0.1
steps = 100
optimal_x = gradient_descent(cost_function_derivative, initial_x, learning_rate, steps)

print(f"Optimal x: {optimal_x:.2f}, Minimum value: {cost_function(optimal_x):.2f}")