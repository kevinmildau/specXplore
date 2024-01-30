# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
import cython
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
from cython cimport boundscheck, wraparound
import copy
import plotly.express as px
from collections import Counter

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_edges_for_selected_above_threshold_from_descending_array(
    signed long long[:] source, signed long long[:] target, double[:] value, signed long long[:] selected_indexes, double threshold):
    """ Cython function loops source target value arrays and extracts those tuples with value above threshold and have
    source or target matching eny entry in selected_indexes. Aborts seach as soon as a value below threshold is 
    encountered. 
    
    IMPORTANT: assumes the value vector and hence the entire edge list to be sorted in descending order. If not the 
    case, the edge filter will stop pre-maturely! """
    assert source.size == target.size == value.size, "Input arrays must be of equal size."
    cdef signed long long max_number_edges = np.int64(source.shape[0])

    cdef set selected_set = set(selected_indexes)

    cdef double[::1] out_value = np.zeros(max_number_edges, dtype=np.double)
    cdef signed long long[::1] out_source = np.zeros(max_number_edges, dtype=np.int64)
    cdef signed long long[::1] out_target = np.zeros(max_number_edges, dtype=np.int64)

    cdef signed long long index
    cdef signed long long counter = 0
    for index in range(0, max_number_edges):
        # OR is used to allow for edges out of the selection set to be saved as well. Those will be visualizaed
        # differently in downstream processing. TODO: improve function name to reflect this.
        if (source[index] in selected_set or target[index] in selected_set) and value[index] > threshold:
            out_value[counter] = value[index]
            out_source[counter] = source[index]
            out_target[counter] = target[index]
            counter += 1
        if value[index] < threshold:
            break
    return np.array(out_value[0:counter]), np.array(out_source[0:counter]), np.array(out_target[0:counter])

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_edges_for_selected_above_threshold_from_descending_array_topk(
    signed long long[:] source, 
    signed long long[:] target, 
    double[:] value, 
    signed long long[:] selected_indexes, 
    double threshold,
    int top_k):
    """ Cython function loops source target value arrays and extracts those tuples with value above threshold and have
    source or target matching eny entry in selected_indexes. In addition, limits the number of occurences of each source
    or target id to the topmost 25 corresponding to the index over the whole edge set. Aborts seach as soon as a value 
    below threshold is encountered. 
    
    IMPORTANT: assumes the value vector and hence the entire edge list to be sorted in descending order. If not the 
    case, the edge filter will stop pre-maturely! """
    assert source.size == target.size == value.size, "Input arrays must be of equal size."
    cdef signed long long max_number_edges = int(source.shape[0])

    cdef set selected_set = set(selected_indexes)

    cdef double[::1] out_value = np.zeros(max_number_edges, dtype=np.double)
    cdef signed long long[::1] out_source = np.zeros(max_number_edges, dtype=np.int64)
    cdef signed long long[::1] out_target = np.zeros(max_number_edges, dtype=np.int64)

    cdef signed long long index
    cdef signed long long counter = 0

    cdef signed long long n_omitted_edges = 0
    cdef max_edge_counter = Counter()
    for index in range(0, max_number_edges):
        if (source[index] in selected_set or target[index] in selected_set) and value[index] > threshold:
            if max_edge_counter[source[index]] >= top_k or max_edge_counter[target[index]] >= top_k:
                # omitt edge and skip adding edge to 
                n_omitted_edges += 1
                continue
            out_value[counter] = value[index]
            out_source[counter] = source[index]
            out_target[counter] = target[index]
            counter += 1
            max_edge_counter.update([source[index]])
            max_edge_counter.update([target[index]])
        if value[index] < threshold:
            break
    return np.array(out_value[0:counter]), np.array(out_source[0:counter]), np.array(out_target[0:counter]), n_omitted_edges

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_edges_above_threshold_from_descending_array(
    signed long long[:] source, 
    signed long long[:] target, 
    double[:] value, 
    double threshold):
    """ Cython function loops source target value arrays and extracts those tuples with value above threshold. Aborts
    seach as soon as a value below threshold is encountered. 
    
    IMPORTANT: assumes the value vector and hence the entire
    edge list to be sorted in descending order. If not the case, the edge filter will stop pre-maturely! 
    """
    #assert source.shape == target.shape == value.shape
    cdef signed long long max_number_edges = int(source.shape[0])
    cdef double[::1] out_value = np.zeros(max_number_edges, dtype=np.double)
    cdef signed long long[::1] out_source = np.zeros(max_number_edges, dtype=np.int64)
    cdef signed long long[::1] out_target = np.zeros(max_number_edges, dtype=np.int64)

    cdef signed long long index
    cdef signed long long counter = 0
    for index in range(0, max_number_edges):
        if value[index] > threshold:
            out_value[counter] = value[index]
            out_source[counter] = source[index]
            out_target[counter] = target[index]
            counter += 1
        if value[index] < threshold:
            break
    return np.array(out_value[0:counter]), np.array(out_source[0:counter]), np.array(out_target[0:counter])