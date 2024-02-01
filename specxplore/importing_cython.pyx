# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
import cython
import numpy as np
from cython cimport boundscheck, wraparound

#@cython.boundscheck(False)
#@cython.wraparound(False)
def construct_unique_pairwise_indices_array(signed long long n_nodes):
    """
    Constructs numpy contiguous array with pairwise numeric indices of form: 
        [
            [value0, value1],
            [value0, value1],
            [value0, value1],
            ...
        ]
    such that all unique pairs of indices are made (order independent, no ordered duplications of pairs).
    """
    cdef signed long long[:, ::1] index_array 
    index_array = np.ascontiguousarray(np.vstack(np.triu_indices(n_nodes, k=1)).T, dtype=np.int64)
    return index_array

#@cython.boundscheck(False)
#@cython.wraparound(False)
def construct_long_format_sim_arrays(
    double[:,::1] similarity_matrix
    ): # beware the tuple output
    """ 
    Constructs unique index pair and value arrays for pairwise similarity matrix.
    
    Returns:
        np.array (signed long long), np.array (signed long long), np.array (double)
    """
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Similarity Matrix provided must be square."
    cdef signed long long n_nodes = np.int64(similarity_matrix.shape[0])
    cdef signed long long[:, ::1] index_array = construct_unique_pairwise_indices_array(n_nodes)
    cdef signed long long n_edges = index_array.shape[0]
    cdef double[::1] value_array = np.zeros(n_edges, dtype=np.double)
    cdef signed long long[::1] source_array = np.zeros(n_edges, dtype=np.int64)
    cdef signed long long[::1] target_array = np.zeros(n_edges, dtype=np.int64)
    cdef signed long long index
    for index in np.arange(0, n_edges, 1, dtype=np.int64):
        value_array[index] = similarity_matrix[index_array[index][0]][index_array[index][1]]
        source_array[index] = index_array[index][0]
        target_array[index] = index_array[index][1]
    return np.array(source_array), np.array(target_array), np.array(value_array)