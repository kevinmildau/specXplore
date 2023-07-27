import numpy as np
import os

similarity_matrix_filepath1 = os.path.join("tests", "similarity_matrix_4_by_4.npy")
similarity_matrix_filepath2 = os.path.join("tests", "similarity_matrix_3_by_3.npy")

# Generate 4 by 4 similarity matrix using numpy random
rng = np.random.default_rng(seed=1)
uniform_matrix = rng.random((4, 4))
np.fill_diagonal(uniform_matrix, 1)
uniform_matrix = np.triu(uniform_matrix)
uniform_matrix = uniform_matrix + uniform_matrix.T - np.diag(np.diag(uniform_matrix))
print(uniform_matrix)
np.save(similarity_matrix_filepath1, uniform_matrix, allow_pickle=False)


fixed_array = np.array([ [1,0.89,0.01],[0.89,1,0.5],[0.01,0.5,1]])
print(fixed_array)
np.save(similarity_matrix_filepath2, fixed_array, allow_pickle=False)