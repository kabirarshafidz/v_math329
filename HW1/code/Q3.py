import numpy as np
import matplotlib.pyplot as plt
from Q2 import funcV, rand_gen_data, f_gradV


def abs_f(theta, t, v, x_dat, y):
    """
    Computes the absolute Taylor remainder for a perturbation t * v:
        R(t) = |f(theta + t*v) - f(theta) - t * <v, nabla f(theta)>|
    
    If the gradient implementation is correct, R(t) = O(t^2) as t -> 0.
    """
    z = theta + t * v
    f_perturbed = funcV(z, x_dat=x_dat, y_label=y)
    f_base = funcV(theta=theta, x_dat=x_dat, y_label=y)
    directional_derivative = np.dot(v, f_gradV(theta=theta, x_dat=x_dat, y_label=y))

    # Absolute difference between actual function change and first-order linear prediction
    pert = np.linalg.norm(f_perturbed - f_base - t * directional_derivative)
    return pert


def plot_():
    """
    Performs the numerical gradient verification on logarithmic axes.
    Generates a log-log plot of the Taylor remainder against step size t,
    fits a line to estimate the asymptotic order, and saves the plot.
    """
    # Generate synthetic dataset and starting point
    X, y, theta = rand_gen_data(n=718, d=10000, seed=42)

    # Sample a random unit perturbation direction: ||v||_2 = 1
    rng = np.random.default_rng(0)
    v = rng.standard_normal(theta.shape)
    v = v / np.linalg.norm(v)

    # Logarithmically spaced step sizes from 10^-8 to 10^0
    t = np.logspace(-8.0, 0.0, num=101)
    
    # Compute remainder for each step size
    rem = []
    for ti in t:
        rem.append(abs_f(theta, ti, v, X, y))
    rem = np.array(rem)

    # Fit straight portion in log-log scale: log10(R(t)) ~ slope * log10(t) + const
    # A slope close to 2 confirms quadratic convergence of the first-order Taylor error
    fit = (t >= 1e-3) & (t <= 1e-1)
    slope = np.polyfit(np.log10(t[fit]), np.log10(rem[fit]), 1)[0]

    # Generate log-log plot
    plt.loglog(t, rem, label="Taylor remainder")
    plt.xlabel("Step size t")
    plt.ylabel(r"$|f(\theta + tv) - f(\theta) - t\langle v, \nabla f(\theta)\rangle|$")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    # Save figure to results directory
    plt.savefig('v_math329/HW1/results/loglogerror.pdf')
    plt.show()

    print(f"slope on 1e-3 <= t <= 1e-1: {slope:.4f}")
