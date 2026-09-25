import numpy as np
import matplotlib.pyplot as plt
from Q2 import funcV, rand_gen_data, f_gradV

def abs_f(theta, t, v, x_dat, y):
    z = theta + t * v
    pert = np.linalg.norm(funcV(z, x_dat=x_dat, y_label=y) - funcV(theta=theta, x_dat=x_dat, y_label=y) - t * np.dot(v, f_gradV(theta=theta, x_dat=x_dat, y_label=y)))
    
    return pert

def plot_():
    X, y, theta = rand_gen_data(n=718, d=10000, seed=42)

    rng = np.random.default_rng(0)
    v = rng.standard_normal(theta.shape)
    v = v / np.linalg.norm(v)

    t = np.logspace(-8.0, 0.0, num=101)
    rem = []
    for ti in t:
        rem.append(abs_f(theta, ti, v, X, y))

    rem = np.array(rem)

    fit = (t >= 1e-3) & (t <= 1e-1)
    slope = np.polyfit(np.log10(t[fit]), np.log10(rem[fit]), 1)[0]

    plt.loglog(t, rem)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('results/loglogerror.pdf')
    plt.show()
    print(f"slope on 1e-3 <= t <= 1e-1: {slope}")

    