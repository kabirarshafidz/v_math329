import timeit
import numpy as np
from load_file import load_data


# ==========================================
# NON-VECTORIZED IMPLEMENTATIONS (FOR-LOOPS)
# ==========================================

def loss_loop(y, X, theta):
    """
    Computes the unregularized smoothed hinge loss using explicit nested loops.
    
    phi(z) =
        0                   if z < -1
        0.5 * (1 + z)^2     if -1 <= z < 0
        0.5 + z             if z >= 0
    where z_i = s_i * <x_i, theta> and s_i = 1 - 2*y_i.
    """
    total = 0.0
    for i in range(len(y)):
        s_i = 1 - 2 * y[i]
        
        # Inner loop: scalar dot product <x_i, theta>
        z_i = 0.0
        for j in range(len(theta)):
            z_i += X[i][j] * theta[j]
        z_i *= s_i

        # Piecewise quadratic smoothing evaluation
        if z_i >= 0:
            total += 0.5 + z_i
        elif z_i >= -1:
            total += 0.5 * (1 + z_i) ** 2

    return total


def grad_loop(y, X, theta):
    """
    Computes the gradient of the unregularized loss using explicit nested loops.
    
    phi'(z) =
        0       if z < -1
        1 + z   if -1 <= z < 0
        1       if z >= 0
    Gradient: sum_i s_i * phi'(z_i) * x_i.
    """
    grad = [0.0] * len(theta)
    
    for i in range(len(y)):
        s_i = 1 - 2 * y[i]
        
        # Inner loop: scalar dot product <x_i, theta>
        z_i = 0.0
        for j in range(len(theta)):
            z_i += X[i][j] * theta[j]
        z_i *= s_i

        # Evaluate scalar derivative phi'(z_i)
        phi_prime = 0.0
        if z_i >= 0:
            phi_prime = 1.0
        elif z_i >= -1:
            phi_prime = 1.0 + z_i

        # Accumulate sample contribution to gradient vector
        for j in range(len(theta)):
            grad[j] += s_i * phi_prime * X[i][j]

    return np.array(grad)


# ==========================================
# VECTORIZED IMPLEMENTATIONS (MATRIX OPS)
# ==========================================

def loss_vectorized(y, X, theta):
    """
    Computes the unregularized smoothed hinge loss across all samples via matrix-vector ops.
    """
    # Vectorized margins: z in R^n where z_i = s_i * <x_i, theta>
    z = (1 - 2 * y) * (X @ theta)
    
    # Piecewise conditional evaluations applied elementwise
    return np.sum(np.where(z >= 0, 0.5 + z,
                  np.where(z >= -1, 0.5 * (1 + z) ** 2, 0.0)))


def grad_vectorized(y, X, theta):
    """
    Computes the gradient of the unregularized loss across all samples via matrix-vector ops.
    """
    s = 1 - 2 * y
    z = s * (X @ theta)
    
    # Sample-wise scalar weights: w_i = s_i * phi'(z_i)
    weights = s * np.where(z >= 0, 1.0, np.where(z >= -1, 1.0 + z, 0.0))
    
    # Matrix product equivalent to sum_i w_i * x_i
    return X.T @ weights


# ==========================================
# REGULARIZED WRAPPERS
# ==========================================

def loss_regularized(y, X, theta, lambda_const, vectorized=True):
    """
    Evaluates regularized objective: f_lambda(theta) = Loss(theta) + (lambda / 2) * ||theta||^2.
    """
    if vectorized:
        return loss_vectorized(y, X, theta) + (lambda_const / 2.0) * np.dot(theta, theta)
        
    return loss_loop(y, X, theta) + (lambda_const / 2.0) * np.dot(theta, theta)


def grad_regularized(y, X, theta, lambda_const, vectorized=True):
    """
    Evaluates regularized gradient: nabla f_lambda(theta) = nabla Loss(theta) + lambda * theta.
    """
    if vectorized:
        return grad_vectorized(y, X, theta) + lambda_const * theta
        
    return grad_loop(y, X, theta) + lambda_const * theta


# ==========================================
# VERIFICATION AND BENCHMARKING
# ==========================================

def verify(y, X, rng, lambda_const=0.005, trials=5, tol=1e-10):
    """
    Verifies numerical agreement up to machine precision between loop and vectorized routines.
    """
    d = X.shape[1]
    rel_losses = []
    rel_grads = []

    for trial in range(trials):
        # Draw random parameter vector from standard Gaussian
        theta = 0.1 * rng.standard_normal(d)

        # Compute values for both implementations
        l_loop = loss_regularized(y, X, theta, lambda_const, vectorized=False)
        l_vec = loss_regularized(y, X, theta, lambda_const, vectorized=True)
        g_loop = grad_regularized(y, X, theta, lambda_const, vectorized=False)
        g_vec = grad_regularized(y, X, theta, lambda_const, vectorized=True)

        # Relative error computations
        rel_loss = abs(l_loop - l_vec) / abs(l_loop)
        rel_losses.append(rel_loss)

        rel_grad = np.linalg.norm(g_loop - g_vec) / np.linalg.norm(g_loop)
        rel_grads.append(rel_grad)

        print(f"trial {trial}: rel loss err {rel_loss:.2e}, rel grad err {rel_grad:.2e}")

    # Summary checks
    print(f"max rel loss err: {max(rel_losses):.2e}")
    print(f"max rel grad err: {max(rel_grads):.2e}")
    print(f"agree within {tol:.0e} on all trials:", max(rel_losses) < tol and max(rel_grads) < tol)


def bench(fn, *args, repeats=5):
    """
    Estimates runtime by taking the minimum execution duration over several repeats.
    """
    return min(timeit.repeat(lambda: fn(*args), number=1, repeat=repeats))


def time_comparison(y, X, rng, lambda_const=0.005):
    """
    Measures and compares runtimes between explicit loops and vectorized routines.
    """
    theta = 0.1 * rng.standard_normal(X.shape[1])

    for name, fn in [("loss", loss_regularized), ("grad", grad_regularized)]:
        t_loop = bench(fn, y, X, theta, lambda_const, False)
        t_vec = bench(fn, y, X, theta, lambda_const, True)
        print(f"{name}: loop {t_loop:.4f}s, vec {t_vec:.4f}s, speedup {t_loop / t_vec:.1f}x")


def run_q2(seed=42):
    """
    Loads dataset, verifies implementation consistency, and profiles run times.
    """
    X, y = load_data(verbose=True)
    rng = np.random.default_rng(seed)

    verify(y, X, rng)
    time_comparison(y, X, rng)


if __name__ == "__main__":
    run_q2()