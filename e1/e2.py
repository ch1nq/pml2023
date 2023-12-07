import sklearn.mixture
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set()

# Construct a 2-component (d=2) GMM (Gaussian mixture model) with
# two outputs (x_1, x_2). Each component has the identity matrix
# I as covariance matrix. The components have means mu_1 = [1, 1]
# and mu_2 = [3, 5]. The mixing coefficients are 1/2.
gmm = sklearn.mixture.GaussianMixture(
    n_components=2,
    weights_init=[0.5, 0.5],
    means_init=[[1, 1], [3, 5]],
    random_state=1337,
    warm_start=True,
)


# Plot samples (n=500) from this density without fitting
def generate_data(n=500):
    x, y = gmm.sample(n)
    return pd.DataFrame({"x_0": x[:, 0], "x_1": x[:, 1], "y": y})


sns.scatterplot(data=generate_data(), x="x_0", y="x_1")
plt.savefig("e2.png")
