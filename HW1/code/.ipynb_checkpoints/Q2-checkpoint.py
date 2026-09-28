import numpy as np
import time


def quad_smooth(x):
    """
    Smoothing function that returns a piece wise quadratic of values
    """
    x = np.array(x, dtype='float')
    z = np.zeros_like(x)
    z[(x >= -1) & (x <= 0)] = 0.5 * (1 + x[(x >= -1) & (x <= 0)]) ** 2
    z[x > 0] = x[x > 0] + 0.5

    return z


def qs_grad(x): 
    """
    Grad of smoothing function
    """
    x = np.array(x, dtype='float')
    z = np.zeros_like(x)
    z[(x >= -1) & (x <= 0)] = 1 + x[(x >= -1) & (x <= 0)]
    z[x > 0] = 1

    return z

###NOT VECTORISED
def funcNv(theta, x_dat, y_label, lam = 0.005):
    total_qs = 0
    for i in range(len(x_dat)):
        s_i = 1 - 2 * y_label[i]
        qs = quad_smooth(s_i * (np.dot(x_dat[i], theta)))

        total_qs += qs

    return total_qs + 0.5 * lam * np.dot(theta, theta)

def f_gradNv(theta, x_dat, y_label, lam=0.005):
    total_qs_grad = np.zeros(len(theta))
    for i in range(len(x_dat)):
        s_i = 1 - 2 * y_label[i]
        c = float(qs_grad(s_i * (x_dat[i] @ theta)))
        total_qs_grad += c * s_i * x_dat[i]
        
    return total_qs_grad + lam * theta

#VECTORISED
def funcV(theta, x_dat, y_label, lam = 0.005):
    s = 1 - 2* y_label
    return np.sum(quad_smooth(s * (x_dat @ theta))) + (lam / 2) * np.linalg.norm(theta) ** 2

def f_gradV(theta, x_dat, y_label, lam=0.005):
    n = len(y_label)
    s = 1 - 2 * y_label
    return lam * theta + x_dat.T @ (qs_grad(s * (x_dat @ theta)) * s)


#VERIFICATION CHECK
def rand_gen_data(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X = np.hstack([X, np.ones((n, 1))])
    y = rng.integers(0, 2, size=n).astype(float)

    theta = rng.standard_normal(d + 1)

    return X, y, theta

def err_check(x, y):
    x = np.asarray(x, dtype='float')
    y = np.asarray(y, dtype='float')

    err = np.linalg.norm(x - y) / max(np.linalg.norm(x), np.linalg.norm(y), 1.0)
    return err

def verification():
    tol = 1e-10

    for i in range(20):
        X, y, theta = rand_gen_data(n=400, d=200, seed=i)
        lam = 0.005

        print("--Running check for f_lamb--")
        a = funcNv(theta, X, y, lam)
        b = funcV(theta, X, y, lam)
        err_f = err_check(a, b)

        if err_f < tol: 
            print(f"Check passed on tol: {tol} with err: {err_f}")

        else:
            print(f"Check failed on tol: {tol} with err: {err_f}")

        print("--Running check for gradient--")
        a = f_gradNv(theta, X, y, lam)
        b = f_gradV(theta, X, y, lam)
        err_f = err_check(a, b)

        if err_f < tol: 
            print(f"Check passed on tol: {tol} with err: {err_f}")

        else:
            print(f"Check failed on tol: {tol} with err: {err_f}")


###TIMING EVERYTHING
def time_(f, *args):
    t0 = time.perf_counter()
    f(*args)
    t1 = time.perf_counter()
    ts = t1 - t0
    return ts

def time_comp():
    t_f_loop  = []
    t_f_vec   = []
    t_g_loop  = []
    t_g_vec   = []

    for i in range(5):
        X, y, theta = rand_gen_data(n=5000, d=200, seed=5)
        lam = 0.0005

        t_f_loop.append(time_(funcNv, theta, X, y, lam))
        t_f_vec.append(time_(funcV, theta, X, y, lam))
        t_g_loop.append(time_(f_gradNv, theta, X, y, lam))
        t_g_vec.append(time_(f_gradV, theta, X, y, lam))

    print(f"f loop: {np.mean(t_f_loop)*1e3} ms, vec: {np.mean(t_f_vec)*1e3} ms, speedup: {np.mean(t_f_loop)/np.mean(t_f_vec)} times")
    print(f"grad loop: {np.mean(t_g_loop)*1e3} ms, vec: {np.mean(t_g_vec)*1e3} ms, speedup: {np.mean(t_g_loop)/np.mean(t_g_vec)} times")