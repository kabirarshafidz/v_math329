# Homework 1

---

## 1. Required Packages (Python)

The codebase requires **Python 3.8+** along with the standard scientific stack:

- `numpy`
- `scipy`
- `matplotlib`

---

## 2. Execution Time

- **Total runtime:** 2m 9.139s

## Results files (in ../results/)

| File | Contents |
|---|---|
| `q3_gradient_check.pdf` | Log-log plot of the Taylor error vs. step size t, with a t² reference line |
| `q3_gradient_check.npz` | Arrays `t` (step sizes) and `error` (the plotted values) |
| `q4_history.csv` | Per-iteration `iteration, f, gnorm`; header comments give the stopping reason and iteration count |
| `q4_GD_(optional).pdf` | Gradient norm vs. iteration |
| `final_theta.npy` | Final iterate θ (vector of length 785) |
| `q5_convergence.pdf` | Objective value and gradient norm vs. iteration (log scale) |
| `q7_error_rates.txt` | `train_error_rate` and `test_error_rate` as fractions |
