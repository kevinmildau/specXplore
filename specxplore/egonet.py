import numpy as np
import dash_cytoscape as cyto
from dash import html
from specxplore import data_transfer_cython, egonet_cython, clustnet_cython
import warnings
import pandas
from typing import List, Dict
import numpy 

# Define Constant: BASIC_NODE_STYLE_SHEET
BASIC_NODE_STYLE_SHEET = [{
    'selector':'node', 
        'style':{
            'height':"10", 'width':'10', 'opacity':0.2, 'content':'data(label)', 'text-halign':'center',
            'text-valign':'center', "shape":"circle", "border-color":"black", "border-width":1}}]
EDGE_STYLE = [{    
    'selector': 'edge',
    'style': {
        'width': 1  # set edge line width to 3
    }}]
# Define Constant: SELECTED_STYLE
SELECTED_STYLE = [{
    'selector': ':selected',
    'style': {
        'background-color': '#30D5C8', 'label': 'data(label)'},
    'selector':'label', 
    'style':{
        'content':'data(label)','color':'black', "font-family": "Ubuntu Mono", "font-size": "1px", "color" : "red",
        "text-wrap": "wrap", "text-max-width": 100,}
    }]

def generate_empty_div_message(plot_type: str) -> html.Div:
    """ Return html container with message specifying that input data is missing for the requested plot.
    Args:
        plot_type (str): Name of the plot that requires generating.
    Returns:
        html.Div: container with message requesting input data for plot_type.
    """
    output = html.Div(html.H6(f"Please provide input data for {plot_type}."))
    return output

# DEVELOPER NOTES:
# candidate for cythonization via turning pandas df into two numpy arrays (x coord, y coord)
# expect minor speed ups since n_nodes < 10_000
#
# Impl. Note: all nodes needed for current implementation of generate_edge_elements_and_styles()
def generate_node_list(data_frame : pandas.DataFrame, mz : numpy.array) -> List[Dict]:
    """ Function generates node list of dictionaries for cytoscape elements.
    
    Args:
        data_frame (pandas.DataFrame): A pandas.DataFrame with numeric columns with x and y axis information for
        nodes.
        mz (numpy.array)

    Returns:
        List[Dict]: A list with dictionary entries for each node giving id, label, and position information.
    """
    number_of_nodes = data_frame.shape[0]
    mz_str = np.array(mz, dtype="str")
    nodes = [{
        'data': {
            'id': str(elem), 
            'label': str(str(elem) + ': ' + mz_str[elem])},
        'position': {
            'x':data_frame["x"].iloc[elem], 
            'y':-data_frame["y"].iloc[elem]},
        'classes':'None'} 
        for elem in range(0, number_of_nodes)]
    return nodes

def construct_cytoscape_egonet(
    elements, style_sheet, boxSelectionEnabled = True, autolock = False, autoungrabify = False, 
    autounselectify = False, userZoomingEnabled = True):
    """ Function returns cytoscape graph object for egonet. """
    cytoscape_graph = cyto.Cytoscape(
        id='cytoscape-tsne-subnet',
        layout={'name':'preset'},
        elements=elements,
        stylesheet=style_sheet,
        boxSelectionEnabled=boxSelectionEnabled, 
        autolock=autolock,  
        autoungrabify=autoungrabify,
        autounselectify=autounselectify, 
        userZoomingEnabled=userZoomingEnabled,
        style={'width':'100%', 'height':'80vh', "border":"1px grey solid","bg":"#feeff4"})
    return cytoscape_graph

def generate_ego_style_selector(ego_id):
    """ Function generates stylesheet selector for ego node in EgoNet"""
    ego_style = [{
        "selector":'node[id= "{}"]'.format(ego_id), 
        "style":{
            "shape":"diamond",'background-color':'gold',
            'opacity':0.9, 'height':'20', 'width':'20', 
            "border-color":"black", "border-width":2}}]
    return ego_style

def construct_ego_net_elements_and_styles(
    data_frame, precursor_masses,  sources, targets, values, threshold, ego_id, expand_level, filter = False):
    """ Function constructs elements for EgoNet cytoscape graph. """
    _,selected_sources, selected_targets = data_transfer_cython.extract_edges_above_threshold(
        sources, targets, values, threshold)
    nodes = generate_node_list(data_frame, precursor_masses)
    bdict = egonet_cython.creating_branching_dict_new(selected_sources, selected_targets, ego_id, int(expand_level))
    edge_elems, edge_styles = egonet_cython.generate_edge_elements_and_styles(
        bdict, selected_sources, selected_targets, nodes)
    # Extract only nodes in connected node set; if deactivated, all nodes shown in cytoscape
    if filter:
        node_ids = set()
        for key in bdict.keys():
            print(bdict[key]["nodes"])
            node_ids.update(set(bdict[key]["nodes"]))
        nodes = [nodes[i] for i in node_ids] 
    # Construct elements list from nodes and edges.
    elements = nodes + edge_elems
    return elements, edge_styles


def generate_egonet_cythonized(
    clust_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, threshold, expand_level):
    max_elements = 20_000
    
    # Check whether a valid cluster selection has been provided, if not return empty div.
    if not clust_selection:
        return generate_empty_div_message("EgoNet")

    # Check whether single node provided, if not, select first node and warn user about selection.
    if len(clust_selection) > 1:
        warnings.warn((
            f"Warning: More than one node selected. Extracting first node {clust_selection[0]}"
            " in input as root for egonet."))
        ego_id = int(clust_selection[0])
    else:
        ego_id = int(clust_selection[0]) # <- DEV NOTE: this always appears as a list. FIX. 
    
    # Construct Data for ego net visualization
    elements, edge_styles = construct_ego_net_elements_and_styles(
        TSNE_DF, MZ, SOURCE, TARGET, VALUE, threshold, ego_id, expand_level)
    
    style_sheet = BASIC_NODE_STYLE_SHEET + edge_styles + generate_ego_style_selector(ego_id) + SELECTED_STYLE + EDGE_STYLE

    # Generate egonet with elements size dependent flexibility
    #if len(elements) <= max_elements:
    #    out = html.Div([construct_cytoscape_egonet(elements, style_sheet)])
    #else:
    #    out = html.Div([construct_cytoscape_egonet(elements, style_sheet, False, True, True, True, False)])
    return elements, style_sheet

