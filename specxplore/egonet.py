import numpy as np
from specxplore import process_matchms as _myfun
import itertools
import dash_cytoscape as cyto
from dash import html
import plotly.express as px
import cython_utils

def generate_egonet_cythonized(
    clust_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, threshold, expand_level):
    
    # Check input length, if not 1, 
    if not clust_selection:
        out = html.Div(html.H6(
            ("Please provide a node selection for EgoNet visualization.")))
        return html.Div(html.H6(["Please provide a node selection for fragmap."]))

    # extract root
    if len(clust_selection) > 1:
        print(("Warning: More than one node selected." +
            "Extracting first node in input as root for egonet." +
            f" Node = {clust_selection[0]}"))
    ego_id = int(clust_selection[0])

    _,s,t = cython_utils.extract_above_threshold(
        SOURCE, TARGET, VALUE, threshold)

    # Not yet cythonized.
    nodes = [{
        'data': {
            'id': str(elem), 
            'label': str(str(elem) + ': ' + str(MZ[elem]))},
        'position': {
            'x':TSNE_DF["x"].iloc[elem], 
            'y':-TSNE_DF["y"].iloc[elem]},
        'classes':'None'} 
        for elem in range(0, TSNE_DF.shape[0])]

    bdict = cython_utils.creating_branching_dict_new(
        s, t, ego_id, int(expand_level))
    edge_elems, edge_styles = cython_utils.generate_edge_elements_and_styles(
        bdict, s, t, nodes)

    elements = nodes + edge_elems
    base_node_style_sheet = [{
            'selector':'node', 
            'style':{
                'height':"100%", 'width':'100%', 'opacity':0.2, 
                'content':'data(label)', 'text-halign':'center',
                'text-valign':'center', "shape":"circle"}}]
    ego_style = [{
        "selector":'node[id= "{}"]'.format(ego_id), 
        "style":{
            "shape":"diamond",'background-color':'gold',
            'opacity':0.95, 'height':'250%', 'width':'250%', 
            "border-color":"black", "border-width":10}}]

    if len(elements) <= 20000:
        out = html.Div([
            cyto.Cytoscape(
                id='cytoscape-tsne-subnet',
                layout={'name':'preset'},
                elements=elements,
                stylesheet= base_node_style_sheet + edge_styles + ego_style,
                boxSelectionEnabled=True,
                style={
                    'width':'100%', 'height':'60vh', "border":"1px grey solid",
                    "bg":"#feeff4"},
            ),
        ])
    else:
        out = html.Div([
            cyto.Cytoscape(
                id='cytoscape-tsne-subnet',
                layout={'name':'preset'},
                elements=elements,
                stylesheet= base_node_style_sheet + edge_styles + ego_style,
                boxSelectionEnabled=False, 
                autolock=True,  
                autoungrabify=True,
                autounselectify=True,
                #panningEnabled=False, 
                userZoomingEnabled=False,
                style={
                    'width':'100%', 'height':'60vh', "border":"1px grey solid",
                    "bg":"#feeff4"},
            ),
        ])
    return out

