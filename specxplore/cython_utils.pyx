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
def construct_unique_pairwise_indices_array(n_nodes):
    cdef long[:, ::1] index_array 
    index_array = np.ascontiguousarray(np.vstack(np.triu_indices(n_nodes, k=1)).T)
    return index_array

#@cython.boundscheck(False)
#@cython.wraparound(False)
def construct_long_format_sim_arrays(double[:,::1] similarity_matrix):
    """ Constructs unqique index pair and value arrays for pairwise similarity matrix 
    
    Returns:
        np.array, np.array, np.array
    """
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Similarity Matrix provided must be square."
    cdef int n_nodes = int(similarity_matrix.shape[0])
    cdef long[:, ::1] index_array = construct_unique_pairwise_indices_array(n_nodes)
    cdef int n_edges = index_array.shape[0]
    cdef double[::1] value_array = np.zeros(n_edges, dtype=np.double)
    cdef long[::1] source_array = np.zeros(n_edges, dtype=np.int_)
    cdef long[::1] target_array = np.zeros(n_edges, dtype=np.int_)
    cdef int index
    for index in range(0, n_edges):
        value_array[index] = similarity_matrix[index_array[index][0]][index_array[index][1]]
        source_array[index] = index_array[index][0]
        target_array[index] = index_array[index][1]
    return np.array(source_array), np.array(target_array), np.array(value_array)

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
    cdef long[::1] out_source = np.zeros(max_number_edges, dtype=np.int_)
    cdef long[::1] out_target = np.zeros(max_number_edges, dtype=np.int_)

    cdef int index
    cdef int counter = 0
    for index in range(0, max_number_edges):
        if source[index] in selected_set and target[index] in selected_set and value[index] > threshold:
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
    cdef long[::1] out_source = np.zeros(max_number_edges, dtype=np.int_)
    cdef long[::1] out_target = np.zeros(max_number_edges, dtype=np.int_)

    cdef int index
    cdef int counter = 0
    for index in range(0, max_number_edges):
        if value[index] > threshold:
            out_value[counter] = value[index]
            out_source[counter] = source[index]
            out_target[counter] = target[index]
            counter += 1
    return np.array(out_value[0:counter]), np.array(out_source[0:counter]), np.array(out_target[0:counter])

#@cython.boundscheck(False)
#@cython.wraparound(False)
def creating_branching_dict_new(long[:] source, long[:] target, long root, long n_levels):
    """Function creates edge branching edge lists.
    
    Assumes all edges defined by source and target pairs are already above threshold!

    Pseudocode:
        Start with Root node id
        Find all edges to it and corresponding nodes. 
        Construct edge and node sets. These are level 0 of branching dictionary.
        For each level in n_levels:
            check for any edges connected to node set from previous level.
            check for any new nodes among those edges.
            Populate branching dict with new edge and node sets.
            Repeat until no new nodes or edges are found, or until n_levels is reached.
    """
    cdef vector[int] edge_ids = np.arange(0, source.shape[0], dtype = np.integer)
    cdef int index
    cdef int inner_index
    cdef set all_nodes = set()
    cdef set all_edges = set()
    cdef int n_edges = source.shape[0]
    cdef set tmp_nodes
    cdef set tmp_edges
    cdef long zero = int(0)
    all_nodes.add(root)
    cdef dict branching_dict = dict()

    # Extract node and edge sets for root node connections
    tmp_nodes = set()
    tmp_edges = set()
    for index in range(zero, n_edges):
        if source[index] == root or target[index] == root:
            tmp_nodes.add(source[index])
            tmp_nodes.add(target[index])
            tmp_edges.add(edge_ids[index]) # or simply j
    
    # Initialize branching dictionary
    if len(tmp_edges) != 0:
        branching_dict[0] = {"nodes": [root], "edges": list(tmp_edges)}
    else:
        branching_dict[0] = {"nodes": [root], "edges": []}    
    
    # Update already covered node and edge sets
    all_edges = all_edges.union(tmp_edges)
    all_nodes = all_nodes.union(tmp_nodes)

    # Expand Branching Dict if possible
    for index in range(1, n_levels):
        if len(tmp_nodes) == zero:
           # print((f"Stopping edge tracing at level:{index}. No new nodes to expand from"))
            break
        current_level_nodes = list(tmp_nodes)
        tmp_nodes = set()
        tmp_edges = set()

        # Find level i edges, and level i+1 nodes
        for inner_index in range(0, n_edges):
            # if the edge connects to any previous nodes, but edge_id isn't captured yet.
            if (source[inner_index] in all_nodes or target[inner_index] in all_nodes) and not (edge_ids[inner_index] in all_edges): # BEWARE OF LONG AND INT TYPING!
                # add edge
                tmp_edges.add(edge_ids[inner_index])
                # add nodes if not yet covered.
                if not source[inner_index] in all_nodes:
                    tmp_nodes.add(source[inner_index])
                if not target[inner_index] in all_nodes:
                    tmp_nodes.add(target[inner_index])

        # Update branching dict with tmp edge and node sets  
        if len(tmp_edges) != zero:
            all_edges = all_edges.union(tmp_edges)
            all_nodes = all_nodes.union(tmp_nodes)
            branching_dict[index] = {
                "nodes": list(current_level_nodes), "edges": list(tmp_edges)}
        else:
            branching_dict[index] = {"nodes": list(current_level_nodes), "edges": list()}
            #print(f"Stopping edge tracing at level:{index}. No new edges found.")
            break
        #branching_dict[index] = {
        #    "nodes": list(current_level_nodes), "edges": list(tmp_edges)}
        #all_edges = all_edges.union(tmp_edges)
        #all_nodes = all_nodes.union(tmp_nodes)
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
    cdef long max_edges = 2500
    cdef long limit_count = 0
    for idx, key in enumerate(branching_dict):
        edge_count += len(branching_dict[key]['edges'])
        node_count += len(branching_dict[key]['nodes'])
    
    edge_elems = [{}] * min(edge_count, max_edges)
    styles = [{}] * (len(branching_dict.keys()))
    
    limit = False
    if edge_count > max_edges:
        print("Branching tree involves more than", str(max_edges), "edges.", 
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


def create_cluster_edge_list(long[:] sources, long[:] targets, long[:] cluster_selection):
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
