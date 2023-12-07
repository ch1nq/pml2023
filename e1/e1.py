import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

sns.set()
np.random.seed(1337)

var = 0.1
X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=20)
y = np.random.normal(loc=X @ np.array([-1, 1]).T, scale=var, size=20)

mean_prior = np.zeros(2)
cov_prior = np.eye(2)
cov_post = np.linalg.inv(np.eye(2) + (1 / var) * (X.T @ X))
mean_post = cov_post @ ((1 / var) * X.T @ y)

print("Mean prior: ", mean_prior)
print("Var  prior: ", cov_prior)
print("Mean posterior: ", mean_post)
print("Var  posterior: ", cov_post)


def plot(mean: np.ndarray, cov: np.ndarray, title: str):
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
    sns.kdeplot(x=X.flatten(), y=Y.flatten(), weights=Z.flatten())
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")


plot(mean_prior, cov_prior, "Prior")
plot(mean_post, cov_post, "Posterior")

plt.savefig("e1.png")
plt.clf()
