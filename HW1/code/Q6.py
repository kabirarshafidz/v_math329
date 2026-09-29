import numpy as np
from scipy.linalg import svdvals
from Q4 import run_GD, data_loader


def compute_L_mu(X, lam):
    """
    Computes Lipschitz smoothness constant L and strong convexity parameter mu.

    Theoretical properties:
      - The unregularized loss has a Lipschitz gradient with constant sigma_max(X)^2
        because |phi''(z)| <= 1 everywhere, implying ||nabla^2 Loss(theta)||_2 <= sigma_max(X)^2.
      - With (lam / 2) * ||theta||^2 regularization:
          L = sigma_max(X)^2 + lam
          mu = lam  (the unregularized loss is convex, so strong convexity comes purely from L2 regularization).
    """
    # Compute only the largest singular value (faster than full SVD)
    # If X is large, svdvals returns singular values in descending order
    sigma_max = svdvals(X)[0]
    L = float(sigma_max**2 + lam)
    mu = float(lam)
    return L, mu


def compare_theory_vs_observed(history, L, mu, lr):
    """
    Compares the theoretical linear convergence rate for L-smooth, mu-strongly
    convex objectives against the observed empirical convergence.

    Theoretical step size condition:
      - Standard theorem requires 0 < gamma < 2 / L (or gamma = 1 / L).
      - For constant step size gamma <= 1 / L, the theoretical convergence factor is:
          rho = 1 - gamma * mu  (or rho = 1 - 1 / kappa with gamma = 1 / L).
    """
    kappa = L / mu
    max_theoretical_step = 2.0 / L

    iters = np.array(history['iteration'])
    f_vals = np.array(history['f'])
    f_star = f_vals[-1]  # Surrogate for optimum based on final iterate

    # Pick a checkpoint halfway through iterations
    k_check = len(iters) // 2
    k_val = iters[k_check]

    # Theoretical rate: rho = 1 - lr * mu (when lr <= 1/L)
    # Using rho = 1 - 1/kappa assumes the canonical choice lr = 1/L
    rho = 1.0 - (lr * mu)
    gap_predicted = (rho**k_val) * (f_vals[0] - f_star)
    gap_observed = f_vals[k_check] - f_star

    print("-" * 55)
    print("QUESTION 6 THEORETICAL ANALYSIS")
    print("-" * 55)
    print(f"Smoothness constant L: {L}")
    print(f"Strong convexity constant mu: {mu}")
    print(f"Condition number kappa (L / mu): {kappa}")
    print(f"Max admissible step size (2 / L): {max_theoretical_step}")
    print(f"Current step size gamma: {lr}")
    print(f"Step size in theoretical bounds: {lr < max_theoretical_step}")
    print("-" * 55)
    print(f"Theoretical convergence factor rho: {rho}")
    print(f"At iteration k = {k_val}:")
    print(f"  Predicted suboptimality gap: {gap_predicted}")
    print(f"  Observed suboptimality gap: {gap_observed:}")
    print("-" * 55)


def q6_compare(path, history=None, lam=0.005, lr=1e-3, tol=1e-3):
    """
    Loads data, extracts smoothness / convexity constants, and evaluates
    empirical vs theoretical convergence rates.
    """
    X_train, _, _, _ = data_loader(path)

    # Avoid rerunning gradient descent if history is already available
    if history is None:
        _, history = run_GD(lr=lr, tol=tol, path=path)

    L, mu = compute_L_mu(X_train, lam)
    compare_theory_vs_observed(history, L, mu, lr)
