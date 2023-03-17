import numpy as np
from specxplore import augmap
import scipy.spatial.distance


nd = 5 * 2 # double of the wanted number of nodes
mock_data = np.random.random(nd).reshape(int(nd/2), 2)
print(mock_data)
norm_data = mock_data / np.linalg.norm(mock_data, 2, axis=1).reshape(-1,1)
sm1 = np.dot(norm_data, norm_data.T)**5 # power to decrease consistently high dot product similarities

# Create dummy matrices
#sm1 = np.repeat([0.9,0.8,0.3,0.1,0.5], 5).reshape([5,5]) # <- not a valid square matrix


#np.fill_diagonal(sm1, 1)
print(sm1)
sm2 = np.repeat([0.6], 5 * 5).reshape([5,5])
np.fill_diagonal(sm2, 1)
sm3 = np.repeat([0.7], 5 * 5).reshape([5,5])
np.fill_diagonal(sm3, 1)

sm2[2][1] = 0.02
sm2[1][2] = 0.02

print(sm2)

sm3[3][2] = 0.01
sm3[2][3] = 0.01

print(sm3)

ids = [1,2,3]
threshold = 0.3

fig = augmap.generate_augmap(ids, sm1, sm2, sm3, threshold)
fig.write_html("tmp_augmap.html")

print("Augmap test run complete, output saved as tmp_augmap.html")