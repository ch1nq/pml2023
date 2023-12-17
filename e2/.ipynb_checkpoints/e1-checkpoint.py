import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import tqdm

np.random.seed(0)
sns.set_theme()
palette = sns.color_palette()


def p_density(x):
    return np.exp(-(x**2) / 2) * (
        np.sin(x) ** 2 + 3 * np.cos(x) ** 2 * np.sin(7 * x) ** 2 + 1
    )


def rejection_sample(q_density, x_func, k=1, n_samples=1000) -> np.ndarray:
    samples = np.zeros((n_samples))
    for i in range(n_samples):
        ratio = 0
        u = 1
        while ratio < u:
            u = np.random.uniform(0, 1)
            x = x_func()
            ratio = p_density(x) / (k * q_density(x))
        samples[i] = x
    return samples


# Self-normalized importance sampling
def importance_sample(q: scipy.stats.norm, n_samples=1000):
    xs = q.rvs(size=n_samples)
    q_samples = q.pdf(xs)
    p_samples = p_density(xs)
    w = (p_samples / q_samples) / np.sum(p_samples / q_samples)
    return np.sum(w * (xs**2))


k_uniform = 23
k_normal = 10

# Estimate E[x^2] by rejection sample x in in [-3, 3]
methods = {
    "RS normal": lambda n: rejection_sample(
        scipy.stats.norm.pdf, np.random.normal, k=k_normal, n_samples=n
    ),
    "RS uniform": lambda n: rejection_sample(
        lambda x: scipy.stats.uniform.pdf(x, loc=-3, scale=6),
        lambda: np.random.uniform(-3, 3),
        k=k_uniform,
        n_samples=n,
    ),
}


def plot_densities():
    method_name, method = list(methods.items())[1]
    xs = np.linspace(-3, 3, 1000)
    sns.lineplot(
        x=xs,
        y=scipy.stats.norm.pdf(xs) * k_normal,
        label=f"Normal envelope ($k={k_normal}$)",
    )
    sns.lineplot(
        x=xs,
        y=scipy.stats.uniform.pdf(xs, loc=-3, scale=6) * k_uniform,
        label=f"Uniform envelope ($k={k_uniform}$)",
    )
    sns.lineplot(
        x=np.linspace(-3, 3, 1000),
        y=p_density(np.linspace(-3, 3, 1000)),
        label="$p(x)$",
        color=palette[1],
    )
    samples = method(10000)
    sns.histplot(samples, stat="density", label="Samples", bins=100, color=palette[4])
    plt.savefig("figs/e1-reject.png")
    plt.clf()


# plot_densities()

n_estimates = 10
data = pd.DataFrame(columns=["proposal", "n_samples", "estimate"])
for _ in tqdm.trange(n_estimates):
    for n_samples in (10, 100, 1000):
        for name, sample_method in methods.items():
            samples = sample_method(n_samples)
            estimate = np.mean(samples**2)
            row = pd.DataFrame(
                {"proposal": name, "n_samples": n_samples, "estimate": estimate},
                index=[0],
            )
            data = pd.concat([data, row], ignore_index=True)
        row = pd.DataFrame(
            {
                "proposal": "IS normal",
                "n_samples": n_samples,
                "estimate": importance_sample(scipy.stats.norm, n_samples=n_samples),
            },
            index=[0],
        )
        data = pd.concat([data, row], ignore_index=True)

actual_mean = np.mean([p_density(x) * x**2 for x in (np.linspace(-3, 3, 10_000))])

sns.barplot(data=data, x="n_samples", y="estimate", hue="proposal", errorbar="sd")
plt.axhline(actual_mean, color="black", linestyle="--", label="Actual")
plt.xlabel("Number of samples")
plt.ylabel("$E[x^2]$")

plt.savefig("figs/e1.png")
plt.clf()

print(data.groupby(["proposal", "n_samples"]).mean())
print(data.groupby(["proposal", "n_samples"]).std())
print("Actual:", actual_mean)

print("IS: ", importance_sample(scipy.stats.norm))
