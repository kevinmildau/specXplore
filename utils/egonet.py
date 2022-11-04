# TODO: add dynamic expansion levels
# TODO: for low numbers of expansion levels, make use of stronger colors
# TODO: make the ego larger and more clearly visible.

import numpy as np
from utils import utils054
from utils import process_matchms as _myfun
import itertools
import dash_cytoscape as cyto
from dash import html


def generate_egonet(clust_selection, SM_MS2DEEPSCORE, TSNE_DF, threshold, expand_level):
    print("Cluster Selection:", clust_selection)
    if len(clust_selection) > 1:
        print("Warning: extracted first node for ego net construction: ", clust_selection[0])
    clust_selection = clust_selection[0]
    # Define Network Data
    n_nodes = SM_MS2DEEPSCORE.shape[0] # all nodes
    adj_m = _myfun.compute_adjacency_matrix(SM_MS2DEEPSCORE, threshold) # <-------------------- EXPENSIVE TO RUN EVERYTIME, FOR FULL DATASET; REQUIRED FOR DYNAMIC THRESHOLD AND EXPAND
    all_possible_edges = list(itertools.combinations(np.arange(0, n_nodes), 2))
    edges = [{'data' : {'id': str(elem[0]) + "-" + str(elem[1]), 
                        'source': str(elem[0]),
                        'target': str(elem[1])}} 
                        for elem in all_possible_edges if (adj_m[elem[0]][elem[1]] != 0)]
    edge_list = [(str(elem[0]), str(elem[1])) for elem in all_possible_edges if (adj_m[elem[0]][elem[1]] != 0)]                    
    edge_dict = utils054.create_edge_dict(edge_list)

    nodes = [{'data': {'id': str(elem), 'label': str('Node ' + str(elem))},
        'position': {'x': TSNE_DF["x"].iloc[elem], 'y': -TSNE_DF["y"].iloc[elem]} # <-- Added -y_coord for correct orientation in line with t-sne
        } 
        for elem in np.arange(0, n_nodes)]
    node_list = [str(elem) for elem in np.arange(0, 1267)]
    node_dict = utils054.create_node_dict(node_list, edge_dict)

    print(len(nodes))
    active_style_sheet = [{'selector' : 'node', 'style' : {'height' : "100%", 
                                                           'width' : '100%', 
                                                           'opacity' : 0.8,
                                                           'content': 'data(label)',
                                                           'text-halign':'center',
                                                           'text-valign':'center',
                                                           #'width':'label',
                                                           #'height':'label',
                                                           "shape" : "circle"}}]
    
    
    elements = nodes # initialize elements to just the nodes
    ego = str(clust_selection)
    bdic = utils054.creating_branching_dict(ego, edge_dict, node_dict, n_levels = expand_level)

    tmp_edge_elems, tmp_edge_styles = utils054.generate_edge_elements_and_styles(bdic, edge_dict)
    elements = elements + tmp_edge_elems
    active_style_sheet = active_style_sheet + tmp_edge_styles + [{"selector": 'node[id= "{}"]'.format(ego), 
        "style": {"shape": "diamond",'background-color': 'gold' ,'opacity': 0.95, 'height': '250%', 'width' : '250%', "border-color": "black", "border-width" : 10}}]

    out = html.Div([
        cyto.Cytoscape(
            id='cytoscape-tsne-subnet',
            layout={'name': 'preset'},
            elements= elements,
            stylesheet=active_style_sheet,
            boxSelectionEnabled=True,
            style={'width': '100%', 'height': '60vh', "border":"1px grey solid", "bg" : "#feeff4"},
        ),
    ])
    return out