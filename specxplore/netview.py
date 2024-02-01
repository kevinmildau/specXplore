import numpy as np
from specxplore import utils_cython, netview_cython, constants

def generate_cluster_node_link_diagram_cythonized(
        TSNE_DF, 
        selected_nodes, 
        selected_class_data, 
        threshold, 
        SOURCE, 
        TARGET, 
        VALUE, 
        is_highlighted, 
        max_edges, 
        max_edges_per_node
        ):
    
    # Extract all nodes and edges connected to the selection
    selected_nodes_np = np.array(selected_nodes, dtype = np.int64)
    
    # Get source and target identifier arrays
    v,s,t, n_omitted_edges_topk = utils_cython.extract_edges_for_selected_above_threshold_from_descending_array_topk(
        SOURCE, TARGET, VALUE, selected_nodes_np, threshold, max_edges_per_node)

    connected_nodes = set(list(np.unique(np.concatenate([s, t]))))             # <---------- for filtering
    connected_nodes.update(set(selected_nodes_np))                             # <---------- for filtering
    
    n_omitted_edges_max_limit = int(0)
    if v.size >= max_edges: # limit size to max_edges
        s = s[0:max_edges]
        t = t[0:max_edges]
        n_omitted_edges_max_limit = v.size - max_edges
        print(f"omitted {n_omitted_edges_max_limit} edges")

    n_omitted_edges = n_omitted_edges_topk + n_omitted_edges_max_limit
    # Create Edge list
    edges = netview_cython.create_edge_list_for_selection(s, t, selected_nodes_np)

    cluster_set = set(selected_nodes)
    n_nodes = TSNE_DF.shape[0]
    nodes = [{}] * n_nodes 
    for i in range(0, n_nodes):
        if (i in cluster_set):
            node_class = selected_class_data[i]
        else:
            node_class = "node_out_of_set"
        
        if is_highlighted[i] == True:
            standard_entry = " is_highlighted"
        else:
            standard_entry = ""
        nodes[i] = {
                'data':{'id':str(i), 
                'label': str(i)},
                'position':{'x':TSNE_DF["x"].iloc[i], 'y':-TSNE_DF["y"].iloc[i]}, 
                'classes': str(node_class) + str(standard_entry)}

    all_styles = (
        constants.GENERAL_STYLE 
        + constants.NETVIEW_STYLE 
        + constants.SELECTED_NODES_STYLE 
        + constants.EDGE_STYLE
    )
    elements = nodes + edges

    return elements, all_styles, n_omitted_edges


