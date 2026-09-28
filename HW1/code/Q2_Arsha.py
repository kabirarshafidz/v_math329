import timeit
import numpy as np
from load_file import load_data


def loss_loop(y, X, theta):
    total = 0
    for i in range(len(y)):
        s_i = 1 - 2 * y[i]
        z_i = s_i * np.dot(X[i], theta)

        if z_i >= 0:
            total += 0.5 + z_i
        elif z_i >= -1:
            total += 0.5 * (1 + z_i) ** 2

    return total


def loss_vectorized(y, X, theta):
    z = (1 - 2 * y) * (X @ theta)
    return np.sum(np.where(z >= 0, 0.5 + z,
                  np.where(z >= -1, 0.5 * (1 + z) ** 2, 0.0)))


def loss_regularized(y, X, theta, lambda_const, vectorized=True):
    if vectorized:
        return loss_vectorized(y, X, theta) + (lambda_const / 2) * np.dot(theta, theta)
        
    return loss_loop(y, X, theta) + (lambda_const / 2) * np.dot(theta, theta)


def grad_loop(y, X, theta):
    grad = np.zeros(len(theta))
    for i in range(len(y)):
        s_i = 1 - 2 * y[i]
        z_i = s_i * np.dot(X[i], theta)

        phi_prime = 0
        if z_i >= 0:
            phi_prime = 1
        elif z_i >= -1:
            phi_prime = 1 + z_i

        grad += s_i * phi_prime * X[i]

    return grad


def grad_vectorized(y, X, theta):
    s = 1 - 2 * y
    z = s * (X @ theta)
    weights = s * np.where(z >= 0, 1, np.where(z >= -1, 1 + z, 0))
    return X.T @ weights


def grad_regularized(y, X, theta, lambda_const, vectorized=True):
    if vectorized:
        return grad_vectorized(y, X, theta) + lambda_const * theta
        
    return grad_loop(y, X, theta) + lambda_const * theta


def verify(y, X, rng, lambda_const=0.005, trials=5, tol=1e-10):
    d = X.shape[1]
    rel_losses = []
    rel_grads = []

    for trial in range(trials):
        theta = 0.1 * rng.standard_normal(d)

        l_loop = loss_regularized(y, X, theta, lambda_const, vectorized=False)
        l_vec = loss_regularized(y, X, theta, lambda_const)
        g_loop = grad_regularized(y, X, theta, lambda_const, vectorized=False)
        g_vec = grad_regularized(y, X, theta, lambda_const)

        rel_loss = abs(l_loop - l_vec) / abs(l_loop)
        rel_losses.append(rel_loss)

        rel_grad = np.linalg.norm(g_loop - g_vec) / np.linalg.norm(g_loop)
        rel_grads.append(rel_grad)

        print(f"trial {trial}: rel loss err {rel_loss:.2e}, rel grad err {rel_grad:.2e}")

    print(f"max rel loss err: {max(rel_losses):.2e}")
    print(f"max rel grad err: {max(rel_grads):.2e}")
    print(f"agree within {tol:.0e} on all trials:", max(rel_losses) < tol and max(rel_grads) < tol)


def bench(fn, *args, repeats=5):
    return min(timeit.repeat(lambda: fn(*args), number=1, repeat=repeats))


def time_comparison(y, X, rng, lambda_const=0.005):
    theta = 0.1 * rng.standard_normal(X.shape[1])

    for name, fn in [("loss", loss_regularized), ("grad", grad_regularized)]:
        t_loop = bench(fn, y, X, theta, lambda_const, False)
        t_vec = bench(fn, y, X, theta, lambda_const, True)
        print(f"{name}: loop {t_loop:.4f}s, vec {t_vec:.4f}s, speedup {t_loop / t_vec:.1f}x")


def run_q2(seed=42):
    X, y = load_data(verbose=True)
    rng = np.random.default_rng(seed)

    verify(y, X, rng)
    time_comparison(y, X, rng)


if __name__ == "__main__":
    run_q2()
