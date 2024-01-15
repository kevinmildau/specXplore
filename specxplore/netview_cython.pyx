# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
import cython
from cython cimport boundscheck, wraparound

#@cython.boundscheck(False)
#@cython.wraparound(False)
def create_cluster_edge_list(
    signed long long[:] sources, 
    signed long long[:] targets, 
    signed long long[:] selection
    ):
    """ 
    Function extracts all sources and targets for specific selection of ids. Threshold independent.
    """
    assert sources.size == targets.size
    edges = [{}] * sources.size
    cdef set selection_set = set(selection)
    cdef signed long long index
    for index in range(0, sources.size):
        source = sources[index]
        target = targets[index]
        if (source in selection_set and target in selection_set):
            edge_class = str("edge_within_set")
        else:
            edge_class = str("edge_out_of_set")
        edges[index] = {
            'data':{'id':str(source) + "-" + str(target), 
            'source':str(source),
            'target':str(target)},
            'classes':str(edge_class)}
    return(edges)