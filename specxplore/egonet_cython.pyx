# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import cython
import numpy as np
from cython cimport boundscheck, wraparound
import plotly.express as px
from collections import Counter

#@cython.boundscheck(False)
#@cython.wraparound(False)
def creating_branching_dict(
        signed long long[:] source, 
        signed long long[:] target, 
        signed long long root, 
        signed long long n_levels, 
        signed long long max_edges): # Returns: tuple ( Dict, signed long long int)
    """
    Function creates edge branching edge lists and returns the number of removed edges as a integer.
    
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
    
    cdef signed long long[:] edge_ids = np.arange(0, source.shape[0], dtype = np.int64)
    cdef signed long long index
    cdef signed long long inner_index
    cdef set all_nodes = set()
    cdef set all_edges = set()
    cdef signed long long n_edges = source.shape[0]
    cdef set tmp_nodes
    cdef set tmp_edges
    cdef signed long long zero = np.int64(0)
    all_nodes.add(root)
    cdef dict branching_dict = dict()
    cdef max_edge_counter = Counter()

    # Extract node and edge sets for root node connections
    tmp_nodes = set()
    tmp_edges = set()
    for index in range(zero, n_edges):
        if source[index] == root or target[index] == root:
            tmp_nodes.add(source[index])
            tmp_nodes.add(target[index])
            tmp_edges.add(edge_ids[index]) # or simply j
            max_edge_counter.update([source[index]])
            max_edge_counter.update([target[index]])
    
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
           # Stopping edge tracing at level:{index}. No new nodes to expand from"
            break
        current_level_nodes = list(tmp_nodes)
        tmp_nodes = set()
        tmp_edges = set()

        # Find level i edges, and level i+1 nodes
        for inner_index in range(0, n_edges):
            # if the edge connects to any previous nodes, but edge_id isn't captured yet.
            # BEWARE OF LONG AND INT TYPING! These need to match for matching to work.
            if (source[inner_index] in all_nodes or target[inner_index] in all_nodes) and not (edge_ids[inner_index] in all_edges): 
                max_edge_counter.update([source[index]])
                max_edge_counter.update([target[index]])
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
            # Stopping edge tracing at level:{index}. No new edges found.
            break
    # Add nodes added in final level if any
    if len(tmp_nodes) != 0:
        branching_dict[n_levels+1] = {"nodes": list(tmp_nodes), "edges": []}
    
    # Post process branching dict to remove any number of edges exceeding the max edge count (
    # Removal done level by level, within level the implicit weight order of edges is used 
    cdef signed long long edge_counter = 0
    cdef signed long long edges_omitted = 0
    cdef signed long long new_edges
    cdef signed long long allowed_new_edges
    for level in branching_dict.keys():
        new_edges = len(branching_dict[level]["edges"])
        if edge_counter + len(branching_dict[level]["edges"]) > max_edges:
            n_new_edges = len(branching_dict[level]["edges"])
            allowed_new_edges = max_edges - edge_counter
            if allowed_new_edges >= 1:
                branching_dict[level]["edges"] = branching_dict[level]["edges"][0:allowed_new_edges]
            else:
                branching_dict[level]["edges"] = []
        edge_counter += new_edges
    edges_omitted = max(0, edge_counter - max_edges)

    return branching_dict, edges_omitted


def generate_edge_elements_and_styles(
        branching_dict, 
        signed long long[:] source, 
        signed long long[:] target, 
        nodes):
    """ 
    Generates an edge list and style list for a given branching dict.
    """
    n_colors = max(2, len(branching_dict) + 1)
    colors = px.colors.sample_colorscale("Viridis", [n/(n_colors -1) for n in range(n_colors)])
    opacities = np.arange(0.8, 0.4, step = -(0.8 - 0.4) / n_colors)
    
    cdef signed long long idx, key, edge
    cdef signed long long node_count = 0
    cdef signed long long node_counter = 0
    cdef signed long long edge_count = 0
    cdef signed long long edge_counter = 0

    for idx, key in enumerate(branching_dict):
        edge_count += len(branching_dict[key]['edges'])
        node_count += len(branching_dict[key]['nodes'])
    
    edge_elems = [{}] * edge_count
    styles = [{}] * (len(branching_dict.keys()))
    
    for idx, key in enumerate(branching_dict):
        styles[idx] = {
            "selector":'.{}'.format(key), 
            "style":{
                "line-color":colors[idx], 
                'opacity':opacities[idx], 
                "background-color":colors[idx], 
                'border-width':'2'}}
        for edge in branching_dict[key]["edges"]:
            edge_elems[edge_counter] = {'data': {
                    'id':"edge_number_" + str(edge),
                    'source':str(source[edge]) ,
                    'target':str(target[edge])},
                    'classes':str(key)}
            edge_counter += 1
        for node in branching_dict[key]["nodes"]:
            nodes[node]['classes'] = str(key)
    return (edge_elems, styles)