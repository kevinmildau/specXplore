import numpy as np
import dash_cytoscape as cyto
from dash import html
from specxplore import data_transfer_cython, clustnet_cython


NODE_SIZE = "10"
EDGE_SIZE = "1"

SELECTION_STYLE = [{
        "selector": '.edge_within_set', 
        "style":{"line-color":"magenta", "background-color" : "magenta", 'opacity':0.6}}, {
        "selector":'.edge_out_of_set', 
        "style":{
            "line-color":"black", 'opacity':0.4, "background-color":"black",'border-width':EDGE_SIZE, 
            'border-color':'black'}},{
        "selector":'.node_out_of_set', 
        "style":{
            "line-color":"black", 'opacity':0.2, "background-color":"black",'border-width':EDGE_SIZE, 
            'border-color':'black'}}]

GENERAL_STYLE = [{
    'selector':'node', 
    'style': {
        'text-halign':'center', 'text-valign':'center', 'background-color': '#E5E4E2',
        'height':NODE_SIZE, 'width':NODE_SIZE, "border-width":EDGE_SIZE, "opacity":0.7}}, {
    
    'selector':'.none',
    'style':{'color' : 'green'},
    #'selector':'label', 
    #'style':{
    #    'content':'data(label)','color':'black', "font-family": "Ubuntu Mono", "font-size": "1px",
    #    "text-wrap": "wrap", "text-max-width": 100,},
    'selector':'.is_standard', 
    'style': {'shape' : 'diamond', 'opacity':1, 'background-color': '#757573'}}]

EDGE_STYLE = [{    
    'selector': 'edge',
    'style': {
        'width': 1  # set edge line width to 3
    }}]

SELECTED_NODES_STYLE = [{
    'selector': ':selected',
    'style': {
        'background-color': 'magenta', "border-color":"purple", "border-width": 1,
        "border-style": "dashed",}}]

def generate_cluster_node_link_diagram_cythonized(
    TSNE_DF, selected_nodes, SM_MS2DEEPSCORE, selected_class_data, 
    threshold, SOURCE, TARGET, VALUE, MZ, is_standard, max_edges):
    
    # Extract all nodes and edges connected to the selection
    selected_nodes_np = np.array(selected_nodes)
    
    # Get source and target identifier arrays
    #v,s,t = data_transfer_cython.extract_selected_above_threshold(
    #    SOURCE, TARGET, VALUE, selected_nodes_np, threshold)
    
    # Get source and target identifier arrays
    v,s,t = data_transfer_cython.extract_selected_above_threshold_top_k(
        SOURCE, TARGET, VALUE, selected_nodes_np, threshold)

    connected_nodes = set(list(np.unique(np.concatenate([s, t]))))             # <---------- for filtering
    connected_nodes.update(set(selected_nodes_np))                             # <---------- for filtering
    
    n_omitted_edges = int(0)
    if v.size >= max_edges: # limit size to max_edges
        s = s[0:max_edges]
        t = t[0:max_edges]
        n_omitted_edges = v.size - max_edges
        print(f"omitted {n_omitted_edges} edges")

    # Create Edge list
    edges = clustnet_cython.create_cluster_edge_list(s,t,selected_nodes_np)

    cluster_set = set(selected_nodes)
    n_nodes = TSNE_DF.shape[0]
    nodes = [{}] * n_nodes 
    for i in range(0, n_nodes):
        if (i in cluster_set):
            node_class = selected_class_data[i]
        else:
            node_class = "node_out_of_set"
        
        if is_standard[i] == True:
            standard_entry = " is_standard"
        else:
            standard_entry = ""
        nodes[i] = {
                'data':{'id':str(i), 
                'label': str(str(i) + ': ' + str(MZ[i]))},
                'position':{'x':TSNE_DF["x"].iloc[i], 'y':-TSNE_DF["y"].iloc[i]}, 
                'classes': str(node_class) + str(standard_entry)}

    all_styles = GENERAL_STYLE + SELECTION_STYLE + SELECTED_NODES_STYLE + EDGE_STYLE
    elements = nodes + edges

    return elements, all_styles, n_omitted_edges


