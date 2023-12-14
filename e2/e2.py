import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

np.random.seed(0)
sns.set_theme()
palette = sns.color_palette()


def p_density(x):
    return np.exp(-(x**2) / 2) / (
        np.sin(x) ** 2 + 3 * np.cos(x) ** 2 * np.sin(7 * x) ** 2 + 1
    )


def rejection_sample(q_density, k=1, n_samples=1000) -> np.ndarray:
    # us = np.random.uniform(0, 1, size=n_samples)
    # xs = np.random.uniform(-3, 3, size=n_samples)
    # return xs[us < p_density(xs) / (k * q_density(xs))]
    samples = np.zeros((n_samples))
    for i in range(n_samples):
        ratio = 0
        u = 1
        while ratio < u:
            u = np.random.uniform(0, 1)
            # x = np.random.uniform(-3, 3)
            x = np.random.normal()
            ratio = p_density(x) / (k * q_density(x))
        samples[i] = x
    return samples


# Self-normalized importance sampling
def importance_sample(q_density, k=1, n_samples=1000):
    xs = np.random.uniform(-3, 3, size=n_samples)
    q_samples = q_density(xs)
    p_samples = p_density(xs)
    norm_ratio = np.mean(p_samples / q_samples)
    # return norm_ratio *


# Estimate E[x^2] by rejection sample x in in [-3, 3]
methods = {
    "RS normal": lambda n: rejection_sample(scipy.stats.norm.pdf, k=2.5, n_samples=n),
    "RS uniform": lambda n: rejection_sample(
        lambda x: scipy.stats.uniform.pdf(x, loc=-3, scale=6), k=3, n_samples=n
    ),
}


def plot_densities():
    method_name, method = list(methods.items())[0]
    xs = np.linspace(-3, 3, 1000)
    sns.lineplot(
        x=xs, y=scipy.stats.norm.pdf(xs) * 2.5, label="Normal envelope ($k=3$)"
    )
    sns.lineplot(
        x=xs,
        y=scipy.stats.uniform.pdf(xs, loc=-3, scale=6) * 6,
        label="Uniform envelope ($k=3$)",
    )
    sns.lineplot(
        x=np.linspace(-3, 3, 1000),
        y=p_density(np.linspace(-3, 3, 1000)),
        label="$p(x)$",
        color=palette[1],
    )
    samples = method(10000)
    # sns.distplot(samples, label="samples")
    sns.histplot(samples, stat="density", label="Samples", bins=100)
    plt.savefig("e2-reject.png")
    plt.clf()


plot_densities()

n_estimates = 10
data = pd.DataFrame(columns=["proposal", "n_samples", "estimate"])
for name, sample_method in methods.items():
    for n_samples in (10, 100, 1000):
        for _ in range(n_estimates):
            samples = sample_method(n_samples)
            estimate = np.mean(samples**2)
            row = pd.DataFrame(
                {"proposal": name, "n_samples": n_samples, "estimate": estimate},
                index=[0],
            )
            data = pd.concat([data, row], ignore_index=True)

actual_mean = np.mean([p_density(x) * x for x in (np.linspace(-3, 3, 100_000))])

sns.barplot(data=data, x="n_samples", y="estimate", hue="proposal", errorbar="sd")
plt.axhline(actual_mean, color="black", linestyle="--", label="Actual")
plt.xlabel("Number of samples")
plt.ylabel("$E[x^2]$")

plt.savefig("e2.png")
plt.clf()

print(data.groupby(["proposal", "n_samples"]).mean())
print(data.groupby(["proposal", "n_samples"]).std())
print("Actual:", actual_mean)

print("IS: ", importance_sample(scipy.stats.norm.pdf) ** 2)
