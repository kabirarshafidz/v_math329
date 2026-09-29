import os
import numpy as np
import matplotlib.pyplot as plt
from Q4 import run_GD, data_loader


def q5_plot_convergence(path, history=None, lr=1e-3, tol=1e-3):
    if history is None:
        X_train, y_train, X_test, y_test = data_loader(path=path)
        _, history = run_GD(lr=lr, tol=tol, path=path)

    iters = history['iteration']
    f_vals = history['f']
    gnorm_vals = history['gnorm']

    # Dos subgráficas separadas (apiladas verticalmente)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # 1. Valor de la función objetivo
    ax1.semilogy(iters, f_vals, color='blue', label=r'$f_\lambda(\theta_k)$')
    ax1.set_ylabel(r'$f_\lambda(\theta_k)$ (log scale)')
    ax1.set_title('Objective Function Value vs Iteration')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    # 2. Norma del gradiente
    ax2.semilogy(iters, gnorm_vals, color='orange', label=r'$\|\nabla f_\lambda(\theta_k)\|$')
    ax2.set_xlabel('Iteration $k$')
    ax2.set_ylabel(r'$\|\nabla f_\lambda(\theta_k)\|$ (log scale)')
    ax2.set_title('Gradient Norm vs Iteration')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Guarda la figura en ../results resolviendo la ruta correctamente
    save_dir = os.path.join('..', 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'q5_convergence.pdf')
    plt.savefig(save_path)
    plt.close()
