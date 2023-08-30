# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
import cython
from cython cimport boundscheck, wraparound
from libcpp.string cimport string

#@cython.boundscheck(False)
#@cython.wraparound(False)
def create_cluster_edge_list(
        long long[:] sources, 
        long long[:] targets, 
        long long[:] cluster_selection
        ):
    assert sources.size == targets.size
    edges = [{}] * sources.size
    cdef set cluster_set = set(cluster_selection)
    cdef string edge_class
    cdef int index
    for index in range(0, sources.size):
        source = sources[index]
        target = targets[index]
        if (source in cluster_set and target in cluster_set):
            edge_class = str("edge_within_set")
        else:
            edge_class = str("edge_out_of_set")
        edges[index] = {
            'data':{'id':str(source) + "-" + str(target), 
            'source':str(source),
            'target':str(target)},
            'classes':edge_class}
    return(edges)
