import os

import matplotlib.pyplot as plt
import numpy as np

from load_file import load_data, RESULTS_DIR
from v_math329.HW1.code.Q2 import loss_regularized, grad_regularized


def gradient_check(y, X, theta, v, t, lambda_const):
    """
    Computes the absolute Taylor remainder for an array of step sizes t:
        R(t) = |f(theta + t*v) - f(theta) - t * <v, nabla f(theta)>|
    
    For a continuously differentiable function with a Lipschitz gradient,
    Taylor's theorem guarantees R(t) = O(t^2) as t -> 0.
    """
    # Base objective and gradient evaluated at the unperturbed point theta
    f0 = loss_regularized(y, X, theta, lambda_const)
    g0 = grad_regularized(y, X, theta, lambda_const)
    
    # First-order directional derivative: <v, nabla f(theta)>
    directional = v @ g0

    # Vectorized list comprehension evaluating the remainder across all perturbation scales
    return np.array([
        abs(loss_regularized(y, X, theta + t_i * v, lambda_const) - f0 - t_i * directional)
        for t_i in t
    ])


def plot_errors(t, errors):
    """
    Generates and saves the log-log verification plot of Taylor errors vs step size t,
    overlaying an ideal theoretical slope-2 reference line (O(t^2)).
    """
    plt.figure(figsize=(6, 6))
    plt.loglog(t, errors, label="Empirical error")
    
    # Anchor the theoretical slope-2 reference line at t = 1e-3 (well within asymptotic regime)
    i_ref = np.argmin(abs(t - 1e-3))
    plt.loglog(t, errors[i_ref] * (t / t[i_ref]) ** 2, "--", label=r"Reference $\propto t^2$")
    
    plt.xlabel("Step size t")
    plt.ylabel(r"$|f(\theta+tv) - f(\theta) - t\langle v, \nabla f(\theta)\rangle|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    # Save high-resolution vector graphic to results directory
    plt.savefig(os.path.join(RESULTS_DIR, "q3_gradient_check.pdf"))
    plt.close()


def run_q3(seed=42):
    """
    Executes Question 3:
    1. Loads MNIST training data and initializes a reproducible random state.
    2. Draws a random parameter vector theta and a normalized unit direction vector v.
    3. Evaluates Taylor remainders across t in [10^-8, 10^0].
    4. Saves the log-log plot and raw data, and fits the empirical slope.
    """
    X, y = load_data()
    d = X.shape[1]
    rng = np.random.default_rng(seed)

    # 101 points logarithmically spaced between 10^-8 and 10^0
    t = np.logspace(-8.0, 0.0, num=101)
    lambda_const = 0.005

    # Random initial point
    theta = 0.1 * rng.standard_normal(d)

    # Random unit direction vector: ||v||_2 = 1
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)

    # Compute errors across step sizes
    errors = gradient_check(y, X, theta, v, t, lambda_const)
    
    # Export visualization and numerical values required by the prompt
    plot_errors(t, errors)
    np.savez(os.path.join(RESULTS_DIR, "q3_gradient_check.npz"), t=t, error=errors)

    # Fit straight portion in log-log scale (avoids machine precision noise at t < 1e-6)
    mask = (t > 1e-6) & (t < 1e-2)
    slope = np.polyfit(np.log10(t[mask]), np.log10(errors[mask]), 1)[0]
    print(f"slope on 1e-6 < t < 1e-2: {slope:.4f}")


if __name__ == "__main__":
    run_q3()
