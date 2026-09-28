import numpy as np
from Q4 import run_GD, data_loader


def compute_L_mu(X, lam):
    sigma_max = np.linalg.svd(X, compute_uv=False)[0]
    L = sigma_max**2 + lam
    mu = lam
    return L, mu


def compare_theory_vs_observed(history, L, mu):
    kappa = L / mu
    rho = 1 - 1 / kappa

    iters = np.array(history['iteration'])
    f_vals = np.array(history['f'])
    f_star = f_vals[-1]

    k_check = len(iters) // 2

    gap_predicted = rho**iters[k_check] * (f_vals[0] - f_star)
    gap_observed = f_vals[k_check] - f_star

    print(f"kappa = {kappa}, rho = 1-1/kappa = {rho}")
    print(f"at k={iters[k_check]} iterations (halfway through the run):")
    print(f"theorem predicts gap = {gap_predicted}")
    print(f"observed gap = {gap_observed}")


def q6_compare(path, lam=0.005, lr=1e-3, tol=1e-3):
    X_train, y_train, X_test, y_test = data_loader(path)
    _, history = run_GD(lr=lr, tol=tol, path=path)
    L, mu = compute_L_mu(X_train, lam)
    compare_theory_vs_observed(history, L, mu)