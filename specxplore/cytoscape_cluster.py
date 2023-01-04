import numpy as np
from specxplore import process_matchms as _myfun
import itertools
import dash_cytoscape as cyto
from dash import html
import plotly.graph_objects as go
from dash import dcc
import cython_utils
import re

selection_style = [
    {
        "selector": '.edge_within_set', 
        "style":{
            "line-color":"magenta", 
            "background-color" : "magenta",
            'opacity':0.6}}, 
    {
        "selector":'.edge_out_of_set', 
        "style":{
            "line-color":"black", 
            'opacity':0.4,
            "background-color":"black",
            'border-width':'2', 
            'border-color':'black'}},
    {
        "selector":'.node_out_of_set', 
        "style":{
            "line-color":"black", 
            'opacity':0.2,
            "background-color":"black",
            'border-width':'2', 
            'border-color':'black'}}]

general_styles = [{
    'selector':'node', 
    'style': {
        'content':'data(label)','text-halign':'center', 
        'text-valign':'center', "shape":"circle",
        'height':"100%", 'width':'100%',
        "border-color":"black", "border-width":2}}, {
    'selector':'label', 
    'style':{
        'content':'data(label)','color':'magenta', 
        "font-family": "Ubuntu Mono",
        "font-size": "16px",
        "color" : "red",
        "text-wrap": "wrap",
        "text-max-width": 100,}}]

def generate_cluster_node_link_diagram_cythonized(
    TSNE_DF, selected_nodes, SM_MS2DEEPSCORE, selected_class_data, color_dict, 
    threshold, SOURCE, TARGET, VALUE, MZ):
    
    # Extract all nodes and edges connected to the selection
    selected_nodes_np = np.array(selected_nodes)
    
    # Get source and target identifier arrays
    v,s,t = cython_utils.extract_cluster_above_threshold(
        selected_nodes_np, SOURCE, TARGET, VALUE, threshold)
    max_edges = 1000
    if v.size >= max_edges: # limit size to max_edges
        indices = np.argsort(v)
        s = s[indices[len(indices)-max_edges:len(indices)]]
        t = t[indices[len(indices)-max_edges:len(indices)]]

    # Create Edge list
    edges = cython_utils.create_cluster_edge_list(s,t,selected_nodes_np)

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

    #all_styles = style_sheet_classes + selection_style
    all_styles = general_styles + selection_style + style_sheet_classes

    out = html.Div([cyto.Cytoscape(
        id='cytoscape-tsne-subnet', layout={'name':'preset'},
        elements=nodes+edges, stylesheet=all_styles,
        boxSelectionEnabled=True,
        style={'width':'100%', 'height':'60vh', 
            "border":"1px grey solid", "bg":"#feeff4"},)])
    return out


