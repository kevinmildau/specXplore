# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
import cython
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
from cython cimport boundscheck, wraparound
import copy
import plotly.express as px

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_selected_above_threshold(
    long[:] source, long[:] target, double[:] value, long[:] selected_indexes, double threshold):
    """ Loops through similarity arrays and filters down to edges for which both nodes are in the selected indexes array 
    and are above threshold. """
    assert source.size == target.size == value.size, "Input arrays must be of equal size."
    cdef int max_number_edges = int(source.shape[0])

    cdef set selected_set = set(selected_indexes)

    cdef double[::1] out_value = np.zeros(max_number_edges, dtype=np.double)
    cdef long[::1] out_source = np.zeros(max_number_edges, dtype=np.int64)
    cdef long[::1] out_target = np.zeros(max_number_edges, dtype=np.int64)

    cdef int index
    cdef int counter = 0
    for index in range(0, max_number_edges):
        # OR is used to allow for edges out of the selection set to be saved as well. Those will be visualizaed
        # differently in downstream processing. TODO: improve function name to reflect this.
        if (source[index] in selected_set or target[index] in selected_set) and value[index] > threshold:
            out_value[counter] = value[index]
            out_source[counter] = source[index]
            out_target[counter] = target[index]
            counter += 1
    return np.array(out_value[0:counter]), np.array(out_source[0:counter]), np.array(out_target[0:counter])

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_edges_above_threshold(
    long[:] source, long[:] target, double[:] value, double threshold):
    """ Cython function loops through similarity list and filters down to
    selection set. """
    #assert source.shape == target.shape == value.shape
    cdef int max_number_edges = int(source.shape[0])
    cdef double[::1] out_value = np.zeros(max_number_edges, dtype=np.double)
    cdef long[::1] out_source = np.zeros(max_number_edges, dtype=np.int64)
    cdef long[::1] out_target = np.zeros(max_number_edges, dtype=np.int64)

    cdef int index
    cdef int counter = 0
    for index in range(0, max_number_edges):
        if value[index] > threshold:
            out_value[counter] = value[index]
            out_source[counter] = source[index]
            out_target[counter] = target[index]
            counter += 1
    return np.array(out_value[0:counter]), np.array(out_source[0:counter]), np.array(out_target[0:counter])