# %%
import pyro
import pyro.infer
import pyro.contrib.gp
import pyro.distributions as dist
import torch
import sklearn.model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

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


def get_data():
    X, y = sample_data_from(g)
    X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, y, test_size=10, random_state=1337)
    return X, X_test, y, y_test


pyro.clear_param_store()

# implement a standard GP model using the maximum a-posteriori estimate of hyper
# parameters and compare that to the sampled GP using NUTS


# Select a suitable model with your own choice of kernel. Identify the parameters
# of the model and decide which parameters are fixed and which are variable
# (you need ≥ 2 variable parameters). We will refer to the variable parameters as theta.
# For each parameter, pick a suitable prior distribution and implement the model
# (or use the GP implemented in Pyro) as well as a function implementing log p(y, θ|X).
def model(X, y=None) -> pyro.contrib.gp.models.GPRegression:
    period = pyro.sample("period", dist.Uniform(0.3, 2))
    lengthscale = pyro.sample("lengthscale", dist.Normal(0.5, 10))
    kernel = pyro.contrib.gp.kernels.Sum(
        pyro.contrib.gp.kernels.Periodic(input_dim=1, period=period),
        pyro.contrib.gp.kernels.RBF(input_dim=1, lengthscale=lengthscale),
    )
    return pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01))


def plot_gp(gp, name):
    Xnew = torch.linspace(0.01, 1, 100)
    means, covs = gp(Xnew, full_cov=True)
    means = means.detach().numpy()
    covs = covs.detach()
    sd = covs.diag().sqrt().numpy()

    # plot the posterior mean and variance of the GP
    sns.lineplot(x=Xnew, y=means, label="mean", color="black")
    plt.fill_between(Xnew, means - sd, means + sd, alpha=0.3, color="black", label="standard deviation")
    sns.scatterplot(x=X, y=y, label="training data", color="red")
    sns.scatterplot(x=X_test, y=y_test, label="test data", color="orange")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name)
    plt.show()


def evaluate(gp: pyro.contrib.gp.models.GPRegression, X_test, y_test):
    # evaluate the posterior log-likelihood of the test set on the fitted GP using θ
    means, covs = gp(X_test, full_cov=True)
    log_likelihood = dist.MultivariateNormal(means, covs).log_prob(y_test)
    return log_likelihood


def task2(X, y, X_test, y_test):
    pyro.clear_param_store()
    # Use SVI to fit the model to the training data
    gp = model(X, y)
    losses = pyro.contrib.gp.util.train(gp, num_steps=1000)

    # Plot the loss
    sns.lineplot(x=range(len(losses)), y=losses)
    plt.show()

    # evaluate the posterior log-likelihood of the test set on the fitted GP using θ*
    # plot_gp(gp, "Task 2")

    print("posterior log-likelihood of the test set", evaluate(gp, X_test, y_test).item())


def compute_log_likelihood(posterior_samples, X_test, y_test):
    likelihoods = []
    for lengthscale, period in zip(posterior_samples["lengthscale"], posterior_samples["period"]):
        with pyro.plate("data"):
            kernel = pyro.contrib.gp.kernels.Sum(
                pyro.contrib.gp.kernels.Periodic(input_dim=1, period=period),
                pyro.contrib.gp.kernels.RBF(input_dim=1, lengthscale=lengthscale),
            )
            gp = pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01))
            likelihoods.append(evaluate(gp, X_test, y_test).item())
    return torch.tensor(likelihoods)


def task3(X, y, X_test, y_test):
    pyro.clear_param_store()
    # Use NUTS to sample from the posterior. Check the quality of the MCMC
    # sampling using diagnostics (Arviz). Use the diagnostics to choose the
    # hyperparameters of the sampling (such as the number of warmup samples).

    nuts_kernel = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=100, num_chains=2, warmup_steps=200)
    mcmc.run(X, y)

    posterior_samples = mcmc.get_samples()

    # Plot the posterior mean and variance of the GP
    lengthscale = posterior_samples["lengthscale"].mean()
    period = posterior_samples["period"].mean()
    kernel = pyro.contrib.gp.kernels.Sum(
        pyro.contrib.gp.kernels.Periodic(input_dim=1, period=period),
        pyro.contrib.gp.kernels.RBF(input_dim=1, lengthscale=lengthscale),
    )
    gp = pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01), jitter=1e-3)
    # plot_gp(gp, "Task 3")

    # Compute the posterior log-likelihood of the test set
    log_likelihoods = compute_log_likelihood(posterior_samples, X_test, y_test)
    print(log_likelihoods.shape)
    print(torch.mean(log_likelihoods))
    print(torch.std(log_likelihoods))


if __name__ == "__main__":
    X, X_test, y, y_test = get_data()
    # task2(X, y, X_test, y_test)
    task3(X, y, X_test, y_test)


# %%
