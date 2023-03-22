import numpy as np

window = np.array([1, 2, None, 1, None, None, 3])
counts = np.unique(window, return_counts=True)
print(counts)
# Output: (array([1, 2, 3, None], dtype=object), array([2, 1, 1, 3]))
