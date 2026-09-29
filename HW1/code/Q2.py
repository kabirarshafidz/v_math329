import numpy as np
import time


def quad_smooth(x):
    """
    Evaluates the piecewise quadratic smoothed hinge loss phi(x).
    
    phi(x) = 
        0                   if x < -1
        0.5 * (1 + x)^2     if -1 <= x <= 0
        x + 0.5             if x > 0
    """
    x = np.array(x, dtype='float')
    z = np.zeros_like(x)
    # Quadratic transition region: smooth approximation between flat and linear parts
    mask_mid = (x >= -1) & (x <= 0)
    z[mask_mid] = 0.5 * (1 + x[mask_mid]) ** 2
    # Linear region: slope of 1
    z[x > 0] = x[x > 0] + 0.5

    return z


def qs_grad(x): 
    """
    Evaluates the derivative of the smoothed hinge loss phi'(x).
    
    phi'(x) = 
        0       if x < -1
        1 + x   if -1 <= x <= 0
        1       if x > 0
    """
    x = np.array(x, dtype='float')
    z = np.zeros_like(x)
    # Derivative in the quadratic region
    mask_mid = (x >= -1) & (x <= 0)
    z[mask_mid] = 1 + x[mask_mid]
    # Derivative in the linear region
    z[x > 0] = 1

    return z


# ==========================================
# NON-VECTORIZED IMPLEMENTATIONS (FOR-LOOPS)
# ==========================================

def funcNv(theta, x_dat, y_label, lam=0.005):
    """
    Computes the regularized objective value using an explicit loop over samples.
    
    Objective: f_lambda(theta) = sum_i phi(s_i * <x_i, theta>) + (lam / 2) * ||theta||^2
    where s_i = 1 - 2 * y_i.
    """
    total_qs = 0.0
    for i in range(len(x_dat)):
        # Map label y in {0, 1} to sign s in {+1, -1}
        s_i = 1 - 2 * y_label[i]
        # Linear score for sample i: s_i * <x_i, theta>
        margin = s_i * np.dot(x_dat[i], theta)
        total_qs += quad_smooth(margin)

    # Add L2 regularization penalty
    return total_qs + 0.5 * lam * np.dot(theta, theta)


def f_gradNv(theta, x_dat, y_label, lam=0.005):
    """
    Computes the gradient of the objective using an explicit loop over samples.
    
    Gradient: nabla f_lambda(theta) = sum_i s_i * phi'(s_i * <x_i, theta>) * x_i + lam * theta
    """
    total_qs_grad = np.zeros(len(theta))
    for i in range(len(x_dat)):
        s_i = 1 - 2 * y_label[i]
        margin = s_i * np.dot(x_dat[i], theta)
        # Scalar derivative phi'(s_i * <x_i, theta>)
        c = float(qs_grad(margin))
        # Accumulate contribution to gradient vector
        total_qs_grad += c * s_i * x_dat[i]
        
    # Add L2 regularization gradient
    return total_qs_grad + lam * theta


# ==========================================
# VECTORIZED IMPLEMENTATIONS (MATRIX OPS)
# ==========================================

def funcV(theta, x_dat, y_label, lam=0.005):
    """
    Computes the regularized objective value using vectorized NumPy operations.
    """
    # Vector of signs s in {+1, -1}^n
    s = 1 - 2 * y_label
    # Margins for all samples computed simultaneously: s * (X @ theta)
    margins = s * (x_dat @ theta)
    # Sum of losses plus L2 regularization term
    return np.sum(quad_smooth(margins)) + (lam / 2.0) * np.linalg.norm(theta) ** 2


def f_gradV(theta, x_dat, y_label, lam=0.005):
    """
    Computes the gradient of the objective using vectorized NumPy operations.
    """
    s = 1 - 2 * y_label
    margins = s * (x_dat @ theta)
    # Scalar weight per sample: s_i * phi'(margins_i)
    weights = qs_grad(margins) * s
    # Gradient: X.T @ weights + lam * theta
    return lam * theta + x_dat.T @ weights


# ==========================================
# VERIFICATION AND BENCHMARKING
# ==========================================

def rand_gen_data(n, d, seed):
    """
    Generates synthetic feature data, binary labels, and parameter vectors for testing.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # Append column of ones to account for intercept/bias
    X = np.hstack([X, np.ones((n, 1))])
    y = rng.integers(0, 2, size=n).astype(float)
    theta = rng.standard_normal(d + 1)

    return X, y, theta


def err_check(x, y):
    """
    Computes relative Euclidean error between two values or arrays.
    """
    x = np.asarray(x, dtype='float')
    y = np.asarray(y, dtype='float')

    # Normalize by max norm to prevent division by zero near origin
    return np.linalg.norm(x - y) / max(np.linalg.norm(x), np.linalg.norm(y), 1.0)


def verification():
    """
    Verifies numerical agreement between loop and vectorized implementations across random seeds.
    """
    tol = 1e-10

    for i in range(20):
        X, y, theta = rand_gen_data(n=400, d=200, seed=i)
        lam = 0.005

        # Objective check
        a = funcNv(theta, X, y, lam)
        b = funcV(theta, X, y, lam)
        err_f = err_check(a, b)
        print(f"[{'PASS' if err_f < tol else 'FAIL'}] Objective error: {err_f:.2e}")

        # Gradient check
        a = f_gradNv(theta, X, y, lam)
        b = f_gradV(theta, X, y, lam)
        err_g = err_check(a, b)
        print(f"[{'PASS' if err_g < tol else 'FAIL'}] Gradient error:  {err_g:.2e}")


def time_(f, *args):
    """
    Measures the wall-clock execution time of a single function call using perf_counter.
    """
    t0 = time.perf_counter()
    f(*args)
    return time.perf_counter() - t0


def time_comp():
    """
    Benchmarks and prints execution times and speedups for loop vs vectorized functions.
    """
    t_f_loop, t_f_vec = [], []
    t_g_loop, t_g_vec = [], []

    for i in range(5):
        X, y, theta = rand_gen_data(n=5000, d=200, seed=5)
        lam = 0.0005

        t_f_loop.append(time_(funcNv, theta, X, y, lam))
        t_f_vec.append(time_(funcV, theta, X, y, lam))
        t_g_loop.append(time_(f_gradNv, theta, X, y, lam))
        t_g_vec.append(time_(f_gradV, theta, X, y, lam))

    print(f"f loop: {np.mean(t_f_loop)*1e3:.2f} ms | vec: {np.mean(t_f_vec)*1e3:.2f} ms | speedup: {np.mean(t_f_loop)/np.mean(t_f_vec):.1f}x")
    print(f"grad loop: {np.mean(t_g_loop)*1e3:.2f} ms | vec: {np.mean(t_g_vec)*1e3:.2f} ms | speedup: {np.mean(t_g_loop)/np.mean(t_g_vec):.1f}x")
