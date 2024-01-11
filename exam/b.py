# %%
import pyro
import pyro.infer
import pyro.contrib.gp
import pyro.distributions as dist
import torch
import sklearn.model_selection
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
torch.random.manual_seed(1337)

# %%


# we assume that observations are given by
# y_i = g(x_i) + epsilon
# epsilon ~ N(0, 0.01)
# where the observations are the grid x_i = (i-1)/(ell-1) for i=1,...,ell with ell=30.
def sample_data_from(g, size=30):
    x = torch.linspace(0, 1, size)
    epsilon = torch.randn(size) * 0.01
    y = g(x) + epsilon
    return x, y


def g(x):
    return -torch.sin(6 * torch.pi * x) ** 2 + 6 * x**2 - 5 * x**4 + 3 / 2


X, y = sample_data_from(g)
X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, y, test_size=10, random_state=1337)

# %%
pyro.clear_param_store()

# implement a standard GP model using the maximum a-posteriori estimate of hyper
# parameters and compare that to the sampled GP using NUTS


# Select a suitable model with your own choice of kernel. Identify the parameters
# of the model and decide which parameters are fixed and which are variable
# (you need ≥ 2 variable parameters). We will refer to the variable parameters as theta.
# For each parameter, pick a suitable prior distribution and implement the model
# (or use the GP implemented in Pyro) as well as a function implementing log p(y, θ|X).
def model(X, y) -> pyro.contrib.gp.models.GPRegression:
    period = pyro.sample("period", dist.Uniform(0.1, 2))
    lengthscale = pyro.sample("lengthscale", dist.Uniform(0, 10))
    kernel = pyro.contrib.gp.kernels.Sum(
        pyro.contrib.gp.kernels.Periodic(input_dim=1, period=period),
        pyro.contrib.gp.kernels.RBF(input_dim=1, lengthscale=lengthscale),
    )
    return pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01))


def train(gpr: pyro.contrib.gp.models.GPRegression):
    losses = pyro.contrib.gp.util.train(gpr, num_steps=1000)
    sns.lineplot(x=range(len(losses)), y=losses)
    plt.show()
    return gpr


def evaluate(gp: pyro.contrib.gp.models.GPRegression, X_test, y_test):
    # evaluate the posterior log-likelihood of the test set on the fitted GP using θ
    # from the previous step

    means, covs = gp(X_test, full_cov=True)
    log_likelihood = dist.MultivariateNormal(means, covs).log_prob(y_test)
    return log_likelihood


if __name__ == "__main__":
    gp = model(X, y)
    gp = train(gp)

    # draw samples from the posterior distribution of the GP using NUTS
    Xnew = torch.linspace(0, 1, 100)
    means, covs = gp(Xnew, full_cov=True)
    means = means.detach().numpy()
    covs = covs.detach()
    sd = covs.diag().sqrt().numpy()

    # plot the posterior mean and variance of the GP
    sns.lineplot(x=Xnew, y=means, label="mean", color="black")
    plt.fill_between(Xnew, means - sd, means + sd, alpha=0.3, color="black", label="standard deviation")
    sns.scatterplot(x=X, y=y, label="training data", color="red")
    sns.scatterplot(x=X_test, y=y_test, label="test data", color="orange")

    print("posterior log-likelihood of the test set", evaluate(gp, X_test, y_test).item())
