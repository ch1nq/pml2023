# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.spatial
import scipy.stats
import scipy.optimize as opt

np.set_printoptions(precision=3)


# %%
def gaussian_kernel(X, Xprime, eta=2, **kwargs):
    gamma = eta
    dists = scipy.spatial.distance.cdist(X, Xprime, metric="sqeuclidean")
    return np.exp(-gamma * dists)


def special_kernel(X, Xprime, eta):
    a = eta[0]
    b = eta[1]
    K = (1 + X @ Xprime.T) ** 2 + a * np.multiply.outer(
        np.sin(2 * np.pi * X.reshape(-1) + b), np.sin(2 * np.pi * Xprime.reshape(-1) + b)
    )
    return K


# A)
S = np.linspace(0, 1, 101).reshape((-1, 1))
kernel = gaussian_kernel(S, S, gamma=10)
np.random.normal(loc=0, scale=kernel)


# %%
# load and normalize Mauna Loa data
data = np.genfromtxt("co2_mm_mlo.csv", delimiter=",")
# 10 years of data for learning
X = data[:120, 2] - 1958
y_raw = data[:120, 3]

# Reshape data
X = X.reshape((-1, 1))
y_raw = y_raw.reshape((-1, 1))

y_mean = np.mean(y_raw)
y_std = np.sqrt(np.var(y_raw))
y = (y_raw - y_mean) / y_std

# the next 5 years for prediction
X_predict = data[120:180, 2] - 1958
y_predict = data[120:180, 3]


# %%
# B)


def negLogLikelihood(params, kernel):
    """calculate the negative loglikelihood (See section 6.3 in the lecture notes)"""
    noise_y = params[0]
    eta = params[1:]
    y_var = np.var(noise_y)
    N = len(y)
    S = np.linspace(0, 1, N).reshape((-1, 1))
    K = kernel(S, S, eta=eta)
    return (
        -0.5 * y.T @ np.linalg.inv(y_var * np.eye(N) + K) @ y
        - 0.5 * np.linalg.slogdet(y_var * np.eye(N) + K)[1]
        - 0.5 * N * np.log(np.sqrt(2 * np.pi))
    )


def optimize_params(ranges, kernel, Ngrid):
    opt_params = opt.brute(lambda params: negLogLikelihood(params, kernel), ranges, Ns=Ngrid, finish=None)
    noise_var = opt_params[0]
    eta = opt_params[1:]
    return noise_var, eta


# B) todo: implement the posterior distribution, i.e. the distribution of f^star
def conditional(X: np.ndarray, y, noise_var, eta, kernel):
    # todo: Write the function...
    # See eq. 66 in the lecture notes. Note that there is a small error: Instead of (S) it should be K(S)
    print(X.shape, y.shape)
    X, x_star = X[: len(y)], X[len(y) :]
    mu_star = kernel(X, x_star, eta=eta).T @ np.linalg.inv(kernel(X, X, eta=eta) + noise_var * np.eye(X.shape[0])) @ y
    # mu_star = kernel(X, x_star).T @ np.linalg.inv(kernel(X, eta=eta) + noise_var * np.eye(X.shape[0])) @ y
    sigma_star = kernel(x_star, x_star, eta=eta) - kernel(X, x_star, eta=eta).T @ np.linalg.inv(
        kernel(X, X, eta=eta) + noise_var * np.eye(X.shape[0])
    ) @ kernel(X, x_star, eta=eta)
    return mu_star, sigma_star  # return mean and covariance matrix


# %%
def plot(kernel, name, ranges):
    Ngrid = 10
    noise_var, eta = optimize_params(ranges, kernel, Ngrid)
    print("optimal params:", noise_var, eta)

    prediction_mean_gp, Sigma_gp = conditional(np.vstack((X, X_predict.reshape((-1, 1)))), y, noise_var, eta, kernel)
    var_gp = np.diag(Sigma_gp).reshape(
        (-1, 1)
    )  # We only need the diagonal term of the covariance matrix for the plots.
    # plotting code for your convenience
    plt.figure(dpi=400, figsize=(6, 3))
    plt.plot(X + 1958, y_raw, color="blue", label="training data")
    plt.plot(X_predict + 1958, y_predict, color="red", label="test data")
    yout_m = prediction_mean_gp * y_std + y_mean
    yout_v = var_gp * y_std**2
    plt.plot(X_predict + 1958, yout_m, color="black", label="gp prediction")
    plt.plot(X_predict + 1958, yout_m + 1.96 * yout_v**0.5, color="grey", label="GP uncertainty")
    plt.plot(X_predict + 1958, yout_m - 1.96 * yout_v**0.5, color="grey")
    plt.xlabel("year")
    plt.ylabel("co2(ppm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{name}.pdf")


# %%
# B) todo: use the learned GP to predict on the observations at X_predict
plot(kernel=gaussian_kernel, ranges=((1.0e-4, 10), (1.0e-4, 10)), name="gaussian_kernel")

# %%
plot(kernel=special_kernel, ranges=((1e-3, 10), (1.0e-1, 10), (1.0e-6, 10)), name="special_kernel")
