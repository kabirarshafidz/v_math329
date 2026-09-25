from Q4 import run_GD, data_loader
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(path, history=None, lr=1e-3, tol=1e-8):
    if history == None:
        X_train, y_train, X_test, y_test = data_loader(path=path)

        _, history = run_GD(lr=lr, tol=tol, path=path)

    iters = history['iteration']
    f_vals = history['f']
    gnorm_vals = history['gnorm']

    plt.figure()
    plt.semilogy(iters, f_vals, label=r'$f_\lambda(\theta_k)$')
    plt.semilogy(iters, gnorm_vals, label=r'$\|\nabla f_\lambda(\theta_k)\|$')
    plt.xlabel('iteration k')
    plt.ylabel('value (log scale)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('v_math329/HW1/results/q5_convergence.pdf')
    plt.show()