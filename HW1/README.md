# HW1

## Running

```bash
cd code
python main.py        # runs all questions
python Q2_Arsha.py    # Q2 only
python Q3_Arsha.py    # Q3 only
```

Data is loaded from `data/mnist_train_test.mat` by `code/load_file.py`.

## Results

- `results/q3_gradient_check.pdf`: Q3 gradient-check plot (log-log).
- `results/q3_gradient_check.npz`: numerical values used for the Q3 plot. NumPy archive with arrays
  `t` (step sizes, `np.logspace(-8, 0, 101)`) and `error`
  (`|f(θ+tv) - f(θ) - t⟨v, ∇f(θ)⟩|`). Load with `np.load("results/q3_gradient_check.npz")`.
