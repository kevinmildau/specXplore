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
def construct_long_format_sim_table(double[:,::1] sm):
    """ Cython function which constructs index pair and value arrays for
    pairwise similarity matrix such that each unique pair and value
    combinaiton is available. """
    assert sm.shape[0] == sm.shape[1]
    cdef int n_nodes = int(sm.shape[0])
    cdef int i
    cdef long[:, ::1] index_array = np.ascontiguousarray(
        np.vstack(np.triu_indices(n_nodes, k=1)).T)
    cdef int n = index_array.shape[0]
    cdef double[::1] value_array = np.zeros(n, dtype=np.double)
    cdef long[::1] source_array = np.zeros(n, dtype=np.int_)
    cdef long[::1] target_array = np.zeros(n, dtype=np.int_)
    for i in range(0, n):
        value_array[i] = sm[index_array[i][0]][index_array[i][1]]
        source_array[i] = index_array[i][0]
        target_array[i] = index_array[i][1]
    return np.array(source_array), np.array(target_array), np.array(value_array)

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_selection(
    long[:] source, long[:] target, double[:] value, long[:] selected_indexes, double threshold):
    """ Cython function loops through similarity list and filters down to
    selection set. """
    #assert source.shape == target.shape == value.shape
    cdef int nmax = int(source.shape[0])
    cdef int i
    cdef int k = 0
    cdef set selected_set = set(selected_indexes)

    cdef double[::1] out_value = np.zeros(nmax, dtype=np.double)
    cdef long[::1] out_source = np.zeros(nmax, dtype=np.int_)
    cdef long[::1] out_target = np.zeros(nmax, dtype=np.int_)

    for i in range(0, nmax):
        if source[i] in selected_set and target[i] in selected_set and value[i] > threshold:
            out_value[k] = value[i]
            out_source[k] = source[i]
            out_target[k] = target[i]
            k += 1
    return np.array(out_value[0:k]), np.array(out_source[0:k]), np.array(out_target[0:k])

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_above_threshold(
    long[:] source, long[:] target, double[:] value, double threshold):
    """ Cython function loops through similarity list and filters down to
    selection set. """
    #assert source.shape == target.shape == value.shape
    cdef int nmax = int(source.shape[0])
    cdef int i
    cdef int k = 0
    cdef double[::1] out_value = np.zeros(nmax, dtype=np.double)
    cdef long[::1] out_source = np.zeros(nmax, dtype=np.int_)
    cdef long[::1] out_target = np.zeros(nmax, dtype=np.int_)
    for i in range(0, nmax):
        if value[i] > threshold:
            out_value[k] = value[i]
            out_source[k] = source[i]
            out_target[k] = target[i]
            k += 1
    return np.array(out_value[0:k]), np.array(out_source[0:k]), np.array(out_target[0:k])

#@cython.boundscheck(False)
#@cython.wraparound(False)
def extract_cluster_above_threshold(
    long[:] selection, long[:] source, long[:] target, double[:] value, 
    double threshold):
    """ Cython function loops through similarity list and filters down to
    threshold and within selection set requirement. """
    #assert source.shape == target.shape == value.shape
    cdef int nmax = int(source.shape[0])
    cdef int i
    cdef int k = 0
    cdef selection_set = set(selection)
    cdef double[::1] out_value = np.zeros(nmax, dtype=np.double)
    cdef long[::1] out_source = np.zeros(nmax, dtype=np.int_)
    cdef long[::1] out_target = np.zeros(nmax, dtype=np.int_)
    for i in range(0, nmax):
        if value[i] > threshold and (
            source[i] in selection_set or target[i] in selection_set):
            out_value[k] = value[i]
            out_source[k] = source[i]
            out_target[k] = target[i]
            k += 1
    return np.array(out_value[0:k]), np.array(out_source[0:k]), np.array(out_target[0:k])


#@cython.boundscheck(False)
#@cython.wraparound(False)
def creating_branching_dict_new(long[:] source, long[:] target, long root, long n_levels):
    """Function creates edge branching edge lists."""
    # PSEUDOCODE
    # Start with root id
    # Root is a node.
    #   For root, find all edges that source or target it and add to edges_set.
    #   In addition, make a set of all nodes that are connected to root. 
    #   This will constitute level 1 of the branching network.
    #   Before going into levels loop, check whether anything beyond root was
    #   added.
    # For lvl in n_levels:
    #   Take note of the current edge and node set.
    #   Sweep the edge list for any edges connected to the current node set.
    #   Extract those edges and nodes, save as lvl_n sets, add lvl_n sets
    #   to global sets.
    #   Before going to next level, assess whether any new nodes were found. If
    #   Not, abort.
    # Input
    #   Any pair in source and target will be above threshold and be a 
    #   legitimate new edge if connected to any other nodes.
    
    # Code Notes --------------------------------------------------------------
    # Edge ids can be simple index integers within this function. Each index
    # points to the corresponding source and target entries.

    #cdef vector[string] edge_ids = np.core.defchararray.add(
    #    source.astype(np.str_), np.repeat(np.str_("-"), len(source)))
    #edge_ids = np.core.defchararray.add(edge_ids, target.astype(np.str_))
    cdef vector[int] edge_ids = np.arange(0, source.shape[0], dtype = np.integer)
    cdef int i
    cdef int j
    cdef set all_nodes = set()
    cdef set all_edges = set()
    cdef int n_edges = source.shape[0]
    cdef set tmp_nodes
    cdef set tmp_edges
    cdef long zero = int(0)
    all_nodes.add(root)
    cdef dict branching_dict = dict()

    # Initialize Layer 1
    tmp_nodes = set()
    tmp_edges = set()
    for j in range(zero, n_edges):
        if source[j] == root or target[j] == root:
            tmp_nodes.add(source[j])
            tmp_nodes.add(target[j])
            tmp_edges.add(edge_ids[j]) # or simply j
    if len(tmp_edges) != 0:
        branching_dict[0] = {"nodes": [root], "edges": list(tmp_edges)}
    else:
        branching_dict[0] = {"nodes": [root], "edges": []}
    all_edges = all_edges.union(tmp_edges)
    all_nodes = all_nodes.union(tmp_nodes)

    # Populate Levels
    for i in range(1, n_levels):
        if len(tmp_nodes) == zero:
            print((f"Stopping edge tracing at level:{i}."
                "No new nodes to expand from"))
            break
        current_level_nodes = list(tmp_nodes)
        tmp_nodes = set()
        tmp_edges = set()
        # This finds level i edges, and level i+1 nodes
        for j in range(0, n_edges):
            # if the edge connects to any previous nodes, but itself edge_id isn't captured yet.
            if (source[j] in all_nodes or target[j] in all_nodes) and not (edge_ids[j] in all_edges): # BEWARE OF LONG AND INT TYPING!
                if not source[j] in all_nodes:
                    tmp_nodes.add(source[j])
                if not target[j] in all_nodes:
                    tmp_nodes.add(target[j])
                tmp_edges.add(edge_ids[j]) # or simply j
        
        if len(tmp_edges) != zero:
            all_edges = all_edges.union(tmp_edges)
            all_nodes = all_nodes.union(tmp_nodes)
            branching_dict[i] = {
                "nodes": list(current_level_nodes), "edges": list(tmp_edges)}
        else:
            branching_dict[i] = {
                "nodes": list(current_level_nodes), "edges": list()}
            print(f"Stopping edge tracing at level:{i}. No new edges found.")
            break

        branching_dict[i] = {
            "nodes": list(current_level_nodes), "edges": list(tmp_edges)}
        all_edges = all_edges.union(tmp_edges)
        all_nodes = all_nodes.union(tmp_nodes)
    
    # Add nodes added in final level if any
    if len(tmp_nodes) != 0:
        branching_dict[n_levels+1] = {"nodes": list(tmp_nodes), "edges": []}
    return branching_dict


def generate_edge_elements_and_styles(
    branching_dict, long[:] source, long[:] target, nodes):
    """Generates an edge list and style list for a given branching dict. """
    n_colors = max(2, len(branching_dict) + 1)
    # "ice" might be a good alternative color
    colors = px.colors.sample_colorscale(
        "Viridis", [n/(n_colors -1) for n in range(n_colors)])
    opacities = np.arange(0.8, 0.4, step = -(0.8 - 0.4) / n_colors)
    # widths = np.arange(5, 0.5, step = -(5 - 0.5) / n_colors)
    cdef long idx, key, edge
    cdef long node_count = 0
    cdef long node_counter = 0
    cdef long edge_count = 0
    cdef long edge_counter = 0
    cdef long max_edges = 400
    cdef long limit_count = 0
    for idx, key in enumerate(branching_dict):
        edge_count += len(branching_dict[key]['edges'])
        node_count += len(branching_dict[key]['nodes'])
    
    edge_elems = [{}] * min(edge_count, max_edges)
    styles = [{}] * (len(branching_dict.keys()))
    
    limit = False
    if edge_count > max_edges:
        print("Branching tree involves more than 200 edges.", 
            "Edges only shown for first level to avoid dash-cytoscape overload.")
        limit = True
        

    for idx, key in enumerate(branching_dict):
        styles[idx] = {
            "selector":'.{}'.format(key), 
            "style":{
                "line-color":colors[idx], 
                'opacity':opacities[idx], 
                #'width':widths[idx],
                "background-color":colors[idx], 
                'border-width':'2', 
                'border-color':'black'}}
        if not limit or idx < 1:
            for edge in branching_dict[key]["edges"]:
                if limit_count < max_edges:
                    edge_elems[edge_counter] = {'data': {
                            'id':str(100000 + edge), 
                            'source':str(source[edge]) ,
                            'target':str(target[edge])},
                            'classes':str(key)}
                    edge_counter += 1
                    limit_count += 1
        for node in branching_dict[key]["nodes"]:
            nodes[node]['classes'] = str(key)
    #print("FROM CYTHON", styles)
    return (edge_elems, styles)


def create_cluster_edge_list(long[:] s, long[:] t, long[:] cluster_selection):
    assert s.size == t.size
    edges = [{}] * s.size
    cdef set cluster_set = set(cluster_selection)
    cdef string edge_class
    for i in range(0, s.size):
        source = s[i]
        target = t[i]
        if (source in cluster_set and target in cluster_set):
            edge_class = str("edge_within_set")
        else:
            edge_class = str("edge_out_of_set")
        edges[i] = {
            'data':{'id':str(source) + "-" + str(target), 
            'source':str(source),
            'target':str(target)},
            'classes':edge_class}
    return(edges)
