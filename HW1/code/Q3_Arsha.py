import os

import matplotlib.pyplot as plt
import numpy as np

from load_file import load_data, RESULTS_DIR
from Q2_Arsha import loss_regularized, grad_regularized


def gradient_check(y, X, theta, v, t, lambda_const):
    f0 = loss_regularized(y, X, theta, lambda_const)
    g0 = grad_regularized(y, X, theta, lambda_const)
    directional = v @ g0

    return np.array([
        abs(loss_regularized(y, X, theta + t_i * v, lambda_const) - f0 - t_i * directional)
        for t_i in t
    ])


def plot_errors(t, errors):
    plt.figure(figsize=(6, 6))
    plt.loglog(t, errors, label="error")
    # reference slope-2 line, anchored at t = 1e-3
    i_ref = np.argmin(abs(t - 1e-3))
    plt.loglog(t, errors[i_ref] * (t / t[i_ref]) ** 2, "--", label=r"$\propto t^2$")
    plt.xlabel("t")
    plt.ylabel(r"$|f(\theta+tv) - f(\theta) - t\langle v, \nabla f(\theta)\rangle|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "q3_gradient_check.pdf"))
    plt.close()


def run_q3(seed=42):
    X, y = load_data()
    d = X.shape[1]
    rng = np.random.default_rng(seed)

    t = np.logspace(-8.0, 0.0, num=101)
    lambda_const = 0.005

    theta = 0.01 * rng.standard_normal(d)

    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)

    errors = gradient_check(y, X, theta, v, t, lambda_const)
    plot_errors(t, errors)
    np.savez(os.path.join(RESULTS_DIR, "q3_gradient_check.npz"), t=t, error=errors)

    mask = (t > 1e-6) & (t < 1e-2)   # adjust to where your plot looks straight
    slope = np.polyfit(np.log10(t[mask]), np.log10(errors[mask]), 1)[0]
    print(f"slope on 1e-6 < t < 1e-2: {slope}")


if __name__ == "__main__":
    run_q3()
