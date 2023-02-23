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
        'content':'data(label)','text-halign':'center', 'text-valign':'center', "shape":"circle",
        'height':NODE_SIZE, 'width':NODE_SIZE, "border-color":"black", "border-width":EDGE_SIZE}}, {
    'selector':'label', 
    'style':{
        'content':'data(label)','color':'magenta', "font-family": "Ubuntu Mono", "font-size": "1px", "color" : "red",
        "text-wrap": "wrap", "text-max-width": 100,}}]

EDGE_STYLE = [{    
    'selector': 'edge',
    'style': {
        'width': 1  # set edge line width to 3
    }}]

SELECTED_NODES_STYLE = [{
        'selector': ':selected',
        'style': {'background-color': '#30D5C8','label': 'data(label)' }}]

def generate_cluster_node_link_diagram_cythonized(
    TSNE_DF, selected_nodes, SM_MS2DEEPSCORE, selected_class_data, color_dict, 
    threshold, SOURCE, TARGET, VALUE, MZ):
    
    # Extract all nodes and edges connected to the selection
    selected_nodes_np = np.array(selected_nodes)
    
    # Get source and target identifier arrays
    v,s,t = data_transfer_cython.extract_selected_above_threshold(
        SOURCE, TARGET, VALUE, selected_nodes_np, threshold)
    
    connected_nodes = set(list(np.unique(np.concatenate([s, t]))))             # <---------- for filtering
    connected_nodes.update(set(selected_nodes_np))                             # <---------- for filtering

    max_edges = 2500
    if v.size >= max_edges: # limit size to max_edges
        indices = np.argsort(v)
        s = s[indices[len(indices)-max_edges:len(indices)]]
        t = t[indices[len(indices)-max_edges:len(indices)]]

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
        nodes[i] = {
                'data':{'id':str(i), 
                'label': str(str(i) + ': ' + str(MZ[i]))},
                'position':{'x':TSNE_DF["x"].iloc[i], 'y':-TSNE_DF["y"].iloc[i]}, 
                'classes': node_class}

    all_classes = list(np.unique(selected_class_data))
    style_sheet_classes = [{
        'selector':f".{clust}", 
        'style':{'background-color':f"{color_dict[clust]}", "opacity": 0.8}} 
        for clust in list(all_classes)]

    #all_styles = style_sheet_classes + SELECTION_STYLE
    all_styles = GENERAL_STYLE + SELECTION_STYLE + style_sheet_classes + SELECTED_NODES_STYLE + EDGE_STYLE
    elements = nodes + edges
    #out = html.Div([cyto.Cytoscape(
    #    id='cytoscape-tsne-subnet', layout={'name':'preset'},
    #    elements=nodes+edges, stylesheet=all_styles + SELECTED_NODES_STYLE,
    #    boxSelectionEnabled=True,
    #    style={'width':'100%', 'height':'80vh', 
    #        "border":"1px grey solid", "bg":"#feeff4"},)])
    return elements, all_styles


