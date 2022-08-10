import numpy as np

rng = np.random.default_rng(seed=1701)  # seed for reproducibility

x1 = rng.integers(10, size=6)  # One-dimensional array
x2 = rng.integers(10, size=(3, 4))  # Two-dimensional array
x3 = rng.integers(10, size=(3, 4, 5))  # Three-dimensional array

print (x1)

