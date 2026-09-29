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

| File | Format |
|--- |---|
| `q3_gradient_check.pdf` | Log-log plot of |f(θ+tv) − f(θ) − t⟨v,∇f(θ)⟩| vs. step size t, with a t² reference line. |
| `q3_gradient_check.npz` | NumPy archive (`np.load`) with arrays `t` (101,) = `np.logspace(-8, 0, 101)` and `error` (101,) = the plotted error values, and `theta` (785,) = the point θ used. |
| `q4_history.csv` | Text CSV read with `np.loadtxt(..., delimiter=",")`. The header comment lines start with `#`: `# stopping_reason=[...]` and `# total_iterations=164`. Then there is one row per iteration k = 0..164 (k = 0 is the initial point) with columns `iteration, f, gnorm` (objective value, gradient norm). |
| `q4_GD__optional_.pdf` | Plot of gradient norm vs. iteration for the GD run. |
| `final_theta.npy` | NumPy array (`np.load`), shape (785,), float64: the final iterate of GD. |
| `q5_convergence.pdf` | Two log-scale panels: objective value f_λ(θ_k) and gradient norm ‖∇f_λ(θ_k)‖ vs. iteration. |
| `q7_error_rates.txt` | Plain text, two `key=value` lines: `train_error_rate=...` and `test_error_rate=...` (fractions in [0,1]). |
