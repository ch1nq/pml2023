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
import numpy as np

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


def get_data(seed):
    X, y = sample_data_from(g)
    X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, y, test_size=10, random_state=seed)
    return X, X_test, y, y_test


pyro.clear_param_store()

# implement a standard GP model using the maximum a-posteriori estimate of hyper
# parameters and compare that to the sampled GP using NUTS


def get_kernel(l, v, p):
    return pyro.contrib.gp.kernels.Periodic(input_dim=1, lengthscale=l, variance=v, period=p)


# Select a suitable model with your own choice of kernel. Identify the parameters
# of the model and decide which parameters are fixed and which are variable
# (you need ≥ 2 variable parameters). We will refer to the variable parameters as theta.
# For each parameter, pick a suitable prior distribution and implement the model
# (or use the GP implemented in Pyro) as well as a function implementing log p(y, θ|X).
def model(X, y=None) -> pyro.contrib.gp.models.GPRegression:
    lengthscale = pyro.sample("lengthscale", dist.Normal(3.0, 1.0))
    variance = pyro.sample("variance", dist.Uniform(0.1, 2.0))
    period = pyro.sample("period", dist.Normal(3.0, 1.0))
    kernel = get_kernel(lengthscale, variance, period)
    return pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01))


def plot_gp(gp, name):
    Xnew = torch.linspace(0.01, 1, 100)
    means, covs = gp(Xnew, full_cov=True)
    means = means.detach().numpy()
    covs = covs.detach()
    sd = covs.detach().diag().sqrt().numpy()

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
    gp.eval()
    # evaluate the posterior log-likelihood of the test set on the fitted GP using θ
    means, covs = gp(X_test, full_cov=True)
    log_likelihood = dist.MultivariateNormal(means, covs).log_prob(y_test)
    return log_likelihood


def task2(X, y, X_test, y_test, plot=False):
    pyro.clear_param_store()
    # Use SVI to fit the model to the training data
    gp = model(X, y)
    losses = pyro.contrib.gp.util.train(gp, num_steps=500)

    if plot:
        # Plot the loss
        sns.lineplot(x=range(len(losses)), y=losses)
        plt.show()

        # evaluate the posterior log-likelihood of the test set on the fitted GP using θ*
        plot_gp(gp, "Gradient descent")

    likelihood = evaluate(gp, X_test, y_test)
    # print("posterior log-likelihood of the test set", likelihood)

    return likelihood.detach()


def compute_log_likelihood(posterior_samples, X_test, y_test):
    likelihoods = []
    for kernel_params in zip(*posterior_samples.values()):
        with pyro.plate("data"):
            kernel = get_kernel(*kernel_params)
            gp = pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01))
            likelihoods.append(evaluate(gp, X_test, y_test).item())
    return torch.tensor(likelihoods)


def plot_arviz(mcmc):
    data = az.from_pyro(mcmc)
    # Specify we want 95% credible interval (hdi=high density interval)
    summary = az.summary(data, hdi_prob=0.95)
    print(summary)

    az.plot_posterior(data, hdi_prob=0.95)
    plt.show()
    plt.clf()

    az.plot_trace(data)
    plt.show()


def task3(X, y, X_test, y_test, plot=False):
    pyro.clear_param_store()

    # Use NUTS to sample from the posterior. Check the quality of the MCMC
    # sampling using diagnostics (Arviz). Use the diagnostics to choose the
    # hyperparameters of the sampling (such as the number of warmup samples).
    nuts_kernel = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=300, num_chains=2, warmup_steps=200)
    mcmc.run(X, y)

    # Plot the posterior mean and variance of the GP
    posterior_samples = mcmc.get_samples(num_samples=500)
    if plot:
        plot_arviz(mcmc)
        kernel_params = tuple(torch.mean(posterior_samples[k]) for k in posterior_samples.keys())
        kernel = get_kernel(*kernel_params)
        gp = pyro.contrib.gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.01))
        plot_gp(gp, "MCMC")

    # Compute the posterior log-likelihood of the test set
    log_likelihoods = compute_log_likelihood(posterior_samples, X_test, y_test)
    # print(log_likelihoods.shape)
    # print(torch.mean(log_likelihoods))
    # print(torch.std(log_likelihoods))
    return log_likelihoods.mean()


if __name__ == "__main__":
    likelihoods_gd = []
    likelihoods_mcmc = []
    for i in range(20):
        X, X_test, y, y_test = get_data(i)
        likelihood_gd = task2(X, y, X_test, y_test, plot=True)
        likelihood_mcmc = task3(X, y, X_test, y_test, plot=True)
        likelihoods_gd.append(likelihood_gd.item())
        likelihoods_mcmc.append(likelihood_mcmc.item())

    plt.bar("Gradient descent", np.mean(likelihoods_gd), yerr=np.std(likelihoods_gd))
    plt.bar("MCMC", np.mean(likelihoods_mcmc), yerr=np.std(likelihoods_mcmc))
    plt.
    plt.show()


# %%
    
# plt.bar("Gradient descent", np.sum(likelihoods_gd))
plt.bar("MCMC", np.sum(likelihoods_mcmc))
plt.show()

# %%
