# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
import cython
#from libcpp.string cimport string # <-- TODO CHECK NEEDED
from libcpp.vector cimport vector
import numpy as np
from cython cimport boundscheck, wraparound
import plotly.express as px

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
    cdef vector[int] edge_ids = np.arange(0, source.shape[0], dtype = np.int64)
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
            # BEWARE OF LONG AND INT TYPING!
            if (source[inner_index] in all_nodes or target[inner_index] in all_nodes) and not (edge_ids[inner_index] in all_edges): 
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
                            'id':str(100000 + edge), # TODO: CHECK WHETHER STRING PREFIX WOULD WORK HERE.
                            'source':str(source[edge]) ,
                            'target':str(target[edge])},
                            'classes':str(key)}
                    edge_counter += 1
                    limit_count += 1
        for node in branching_dict[key]["nodes"]:
            nodes[node]['classes'] = str(key)
    #print("FROM CYTHON", styles)
    return (edge_elems, styles)