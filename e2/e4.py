# %% [markdown]
# # Iris data set: inference with NN / SVI solution

# %% [markdown]
#

# %% [markdown]
# First, install the required Python packages on the fly on Colab.

# %% [markdown]
# Import the required Python packages.

# %%
import pyro
import numpy
import torch
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import model_selection
import pyro.distributions as pdist
import torch.distributions as tdist
import torch.nn as tnn
import pyro.nn as pnn
import arviz

# %% [markdown]
# Set some parameters for inference and make reproducible.
#
#

# %%
seed_value = 42  # Replace with your desired seed value
torch.manual_seed(seed_value)
pyro.set_rng_seed(seed_value)
numpy.random.seed(seed_value)

# MAP or diagonal normal?
MAP = True
if MAP:
    MAXIT = 2000  # SVI iterations
    REPORT = 200  # Plot ELBO each time after this amount of SVI iterations
else:
    MAXIT = 100000
    REPORT = 1000

# Number of samples used in prediction
S = 500

# %% [markdown]
# Function to evaluate the accuracy of our trained model.


# %%
def accuracy(pred, data):
    """
    Calculate accuracy of predicted labels (integers).

    pred: predictions, tensor[sample_index, chain_index, data_index, logits]
    data: actual data (digit), tensor[data_index]

    Prediction is taken as most common predicted value.
    Returns accuracy (#correct/#total).
    """
    n = data.shape[0]
    correct = 0
    total = 0
    for i in range(0, n):
        # Get most common prediction value from logits
        pred_i = int(torch.argmax(torch.sum(pred[:, 0, i, :], 0)))
        # Compare prediction with data
        if int(data[i]) == int(pred_i):
            correct += 1.0
        total += 1.0
    # Return fractional accuracy
    return correct / total


# %% [markdown]
# Load the [iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) set from [scikit-learn](https://sklearn.org/).

# %%
# Iris data set
Dx = 4  # Input vector dim
Dy = 3  # Number of labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


iris = sklearn.datasets.load_iris()
x_all = torch.tensor(iris.data, dtype=torch.float)  # Input vector (4D)
y_all = torch.tensor(iris.target, dtype=torch.int)  # Label(3 classes)

x_all = x_all.to(device)
y_all = y_all.to(device)

# Make training and test set
x, x_test, y, y_test = sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.33, random_state=42)

print("Data set / test set sizes: %i, %i." % (x.shape[0], x_test.shape[0]))

# %% [markdown]
# The probabilistic model, implemented as a callable class. We could also simply use a function.
#


# %%
class Model:
    def __init__(self, x_dim=4, y_dim=3, h_dim=5):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim

    def __call__(self, x, y=None):
        """
        We need None for predictive
        """
        x_dim = self.x_dim
        y_dim = self.y_dim
        h_dim = self.h_dim
        # Number of observations
        n = x.shape[0]
        # standard deviation of Normals
        sd = 1  # EXERCISE: 100->1
        # Layer 1
        w1 = pyro.sample("w1", pdist.Normal(0, sd).expand([x_dim, h_dim]).to_event(2))
        b1 = pyro.sample("b1", pdist.Normal(0, sd).expand([h_dim]).to_event(1))
        # Layer 2 # EXERCISE: added layer
        w2 = pyro.sample("w2", pdist.Normal(0, sd).expand([h_dim, h_dim]).to_event(2))
        b2 = pyro.sample("b2", pdist.Normal(0, sd).expand([h_dim]).to_event(1))
        # Layer 3
        w3 = pyro.sample("w3", pdist.Normal(0, sd).expand([h_dim, y_dim]).to_event(2))
        b3 = pyro.sample("b3", pdist.Normal(0, sd).expand([y_dim]).to_event(1))
        # NN
        h1 = torch.tanh((x @ w1) + b1)
        h2 = torch.tanh((h1 @ w2) + b2)  # EXERCISE: added layer
        logits = h2 @ w3 + b3
        # Save deterministc variable (logits) in trace
        pyro.deterministic("logits", logits)
        # Categorical likelihood
        with pyro.plate("labels", n):
            obs = pyro.sample("obs", pdist.Categorical(logits=logits), obs=y)


if __name__ == "__main__":
    # Instantiate the Model object
    model = Model()

    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)

    S = 500  # Number of samples
    W = 100  # Number of warm up samples (tuning of NUTS)
    C = 2  # Number of chains
    nuts_kernel = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=S, num_chains=C, warmup_steps=W)

    print(x.shape)
    print(y.shape)

    # Clear any previously used parameters
    pyro.clear_param_store()
    mcmc.run(x, y)

    # Print the estimated parameters.

    # %%
    # Get the [posterior predictive distribution](https://en.wikipedia.org/wiki/Posterior_predictive_distribution) by sampling the model's parameters from the Guide object and applying the model to the test set.
    posterior_samples = mcmc.get_samples()
    posterior_predictive = pyro.infer.Predictive(model, posterior_samples)(x_test, None)

    # Evaluate the accuracy of the model on the test set.
    # Print accuracy
    logits = posterior_predictive["logits"]
    print(f"Shape of logits: {logits.shape}")
    print(f"Success: {accuracy(logits, y_test):.2f}%")

    # %% Check quality of samples using arviz
    data = arviz.from_pyro(mcmc)
    summary = arviz.summary(data)
    # mean std of r_hat and ess over all parameters
    r_hat_mean = summary["r_hat"].mean().round(2)
    r_hat_std = summary["r_hat"].std().round(2)
    ess_mean = summary["ess_bulk"].mean().round(2)
    ess_std = summary["ess_bulk"].std().round(2)
    print(f"r_hat: {r_hat_mean} +/- {r_hat_std}")
    print(f"ess: {ess_mean} +/- {ess_std}")

    # %%
# %%
