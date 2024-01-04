import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 20 points x ~ N(0, I)
X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=20)

# True parameters
theta_true = np.array([-1, 1])

# Generate labels y according to the distribution p(y|x)
noise_variance = 0.1
y = np.random.normal(X @ theta_true, np.sqrt(noise_variance))

# Prior distribution for theta ~ N(0, I)
prior_mean = np.zeros(2)
prior_covariance = np.eye(2)

# Compute posterior distribution parameters
sigma_squared = noise_variance
posterior_covariance = np.linalg.inv(
    np.linalg.inv(prior_covariance) + (1 / sigma_squared) * X.T @ X
)
posterior_mean = posterior_covariance @ (1 / sigma_squared) * X.T @ y

# Display results
print("Generated dataset:")
print("X:")
print(X)
print("y:")
print(y)

print("\nTrue parameters:")
print("theta_true:", theta_true)

print("\nPosterior distribution:")
print("Mean:", posterior_mean)
print("Covariance matrix:")
print(posterior_covariance)
