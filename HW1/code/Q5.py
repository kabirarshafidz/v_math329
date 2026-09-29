import os
import numpy as np
import matplotlib.pyplot as plt
from Q4 import run_GD, data_loader


def q5_plot_convergence(path, history=None, lr=1e-3, tol=1e-3):
    if history is None:
        _, history = run_GD(lr=lr, tol=tol, path=path)

    iters = history['iteration']
    f_vals = history['f']
    gnorm_vals = history['gnorm']

    # Two separate subplotsd
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Objective function value
    ax1.semilogy(iters, f_vals, color='blue', label=r'$f_\lambda(\theta_k)$')
    ax1.set_xlabel('Iteration $k$')
    ax1.set_ylabel(r'$f_\lambda(\theta_k)$ (log scale)')
    ax1.set_title('Objective Function Value vs Iteration')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    ax1.set_yticks([1e2, 1e3])

    # 2. Gradient norm
    ax2.semilogy(iters, gnorm_vals, color='orange', label=r'$\|\nabla f_\lambda(\theta_k)\|$')
    ax2.set_xlabel('Iteration $k$')
    ax2.set_ylabel(r'$\|\nabla f_\lambda(\theta_k)\|$ (log scale)')
    ax2.set_title('Gradient Norm vs Iteration')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save the figure in ../results resolving the path correctly
    save_dir = os.path.join('..', 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'q5_convergence.pdf')
    plt.savefig(save_path)
    plt.close()
