# Simulated dataset
data = np.random.rand(100, 5)  # 100 samples, 5 features

# Apply PCA to reduce to 2 dimensions
reduced_data = pca(data, n_components=2)

print("Original shape:", data.shape)
print("Reduced shape:", reduced_data.shape)