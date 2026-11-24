import numpy as np

M = np.kron([[1, 1], [0, 1]], np.eye(2))   # 4x4
B = np.kron([[0], [1]], np.eye(2))         # 4x2

def conjugate_gradient(H, b, tol=1e-6):
    v = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = r @ r
    for _ in range(len(b)):
        Hp = H @ p
        alpha = rr / (p @ Hp)
        v = v + alpha * p
        r = r - alpha * Hp
        rr_new = r @ r
        if np.sqrt(rr_new) < tol:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return v

def construct_AC(T=1):
    I2 = np.eye(2)
    A = np.zeros((4*T, 2*T))
    C = np.zeros((4*T, 4))
    for i in range(1, T+1):
        rows = slice(4*(i-1), 4*i)
        C[rows, :] = np.kron([[1, i], [0, 1]], I2)
        for j in range(1, i+1):
            A[rows, slice(2*(j-1), 2*j)] = np.kron([[i-j], [1]], I2)

    return A, C

def mpcfree(z_t, T=1):
    """
    outputs u_t
    """
    # construct A and C
    A, C = construct_AC(T)

    # H and b
    H = A.T @ A
    b = - A.T @ C @ z_t

    # solve for u
    u = conjugate_gradient(H, b, tol=1e-6)

    return u[:2]

def simulate(sigma, T, steps, seed=0):
    rng = np.random.default_rng(seed)
    z = np.array([1, 0, 0, 0.25])
    Z, U = [z], []
    for t in range(steps):
        u = mpcfree(z, T)                                   # solve, keep only the first control
        z = M @ z + B @ u + sigma * rng.standard_normal(4)  # true dynamics WITH noise
        Z.append(z); U.append(u)
    return np.array(Z), np.array(U) 

def main():

    sigma = 0.01
    T = 10
    steps = 200
    return simulate(sigma, T=T, steps=steps, seed=0)
