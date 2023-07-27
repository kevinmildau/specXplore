import numpy as np
import os
from specxplore.augmap import generate_optimal_leaf_ordering_index, extract_sub_matrix, reorder_matrix
import pytest

similarity_matrix_filepath1 = os.path.join("tests", "similarity_matrix_4_by_4.npy")
similarity_matrix_filepath2 = os.path.join("tests", "similarity_matrix_3_by_3.npy")


sim_matrix1 = np.load(similarity_matrix_filepath1)
sim_matrix2 = np.load(similarity_matrix_filepath2)


def test_input_output_optimal_leaf_ordering() -> None:
    # actual optimality of leaf ordering not evaluated here
    assert type(generate_optimal_leaf_ordering_index(sim_matrix1)) == np.ndarray, "output expected to be be ndarray."
    assert type(generate_optimal_leaf_ordering_index(sim_matrix2)) == np.ndarray, "output expected to be be ndarray."
    with pytest.raises(ValueError):
        generate_optimal_leaf_ordering_index(np.int64(1))

def test_submatrix_extraction() -> None:
    test_array = np.array([ [1,0.89,0.01],[0.89,1,0.5],[0.01,0.5,1]])
    assert extract_sub_matrix([0,1], test_array).all() == np.array([ [1,0.89],[0.89,1]]).all()
    assert extract_sub_matrix([0,2], test_array).all() == np.array([ [1,0.01],[0.01,1]]).all()
    assert extract_sub_matrix([0], test_array).all() == np.array([[1]]).all()
    with pytest.raises(IndexError):
        extract_sub_matrix([10], test_array)

def test_reorder_matrix() -> None:
    test_array = np.array([ [1,0.89,0.01],[0.89,1,0.5],[0.01,0.5,1]])
    assert np.array_equal(reorder_matrix([2,1,0], test_array), np.array([[1, 0.5 ,0.01],[0.5,1,0.89],[0.01,0.89,1]]))
    assert np.array_equal(reorder_matrix([2,0,1], test_array), np.array([[1,0.01,0.5], [0.01,1,0.89], [0.5,0.89,1]]))