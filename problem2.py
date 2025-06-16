import numpy as np

# Generate a 1D NumPy array of 20 random floats between 0 and 10
array = np.random.uniform(0, 10, 20)

# Print the array rounded to 2 decimal places
rounded = np.round(array, 2)
print("Rounded Array:\n", rounded)

# Calculate min, max, and median
print("Minimum:", np.min(rounded))
print("Maximum:", np.max(rounded))
print("Median:", np.median(rounded))

# Replace elements < 5 with their square
transformed = np.where(rounded < 5, rounded**2, rounded)
print("Transformed Array:\n", transformed)

# Alternating sort: smallest, largest, 2nd smallest, 2nd largest, ...
def numpy_alternate_sort(array):
    sorted_arr = np.sort(array)
    result = []
    for i in range(len(sorted_arr) // 2):
        result.append(sorted_arr[i])
        result.append(sorted_arr[-(i + 1)])
    if len(sorted_arr) % 2 != 0:
        result.append(sorted_arr[len(sorted_arr) // 2])
    return np.array(result)

print("Alternating Sort:\n", numpy_alternate_sort(rounded))
