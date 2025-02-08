import numpy as np

# Initial conditions
initial_velocity = 50  # m/s
angle = 45  # degrees
g = 9.81  # gravitational acceleration (m/s^2)

# Convert angle to radians and compute velocity components
theta = np.radians(angle)
v_x = initial_velocity * np.cos(theta)
v_y = initial_velocity * np.sin(theta)

# Time of flight
time_of_flight = 2 * v_y / g

# Position function
def position(t):
    x = v_x * t
    y = v_y * t - 0.5 * g * t**2
    return np.array([x, y])

# Compute arc length of the trajectory
def trajectory_derivative(t):
    dx_dt = v_x
    dy_dt = v_y - g * t
    return np.sqrt(dx_dt**2 + dy_dt**2)

arc_length_projectile = arc_length(trajectory_derivative, 0, time_of_flight)
print(f"Arc length of the projectile's trajectory: {arc_length_projectile:.2f} meters")