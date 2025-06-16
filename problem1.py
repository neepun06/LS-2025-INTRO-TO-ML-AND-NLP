import numpy as np

# Step 1: Create random 2D array
np.random.seed(42)
arr = np.random.randint(1, 51, size=(5, 4))
print("Array:\n", arr)

# Step 2: Anti-diagonal (top-right to bottom-left)
anti_diag = [arr[i, -1 - i] for i in range(min(arr.shape))]
print("Anti-diagonal:", anti_diag)

# Step 3: Max of each row
max_per_row = np.max(arr, axis=1)
print("Max in each row:", max_per_row)

# Step 4: Elements ≤ mean
mean_val = np.mean(arr)
filtered_arr = arr[arr <= mean_val]
print("Elements ≤ mean (", mean_val, "):", filtered_arr)

# Step 5: Boundary traversal
def numpy_boundary_traversal(matrix):
    top = matrix[0, :]
    right = matrix[1:-1, -1]
    bottom = matrix[-1, ::-1]
    left = matrix[-2:0:-1, 0]
    return list(top) + list(right) + list(bottom) + list(left)

print("Boundary Traversal:", numpy_boundary_traversal(arr))
