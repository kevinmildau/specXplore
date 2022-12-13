# TODO: add dynamic expansion levels
# TODO: for low numbers of expansion levels, make use of stronger colors
# TODO: make the ego larger and more clearly visible.
import numpy as np
from utils import process_matchms as _myfun
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
    clust_selection = clust_selection[0]

    v,s,t = cython_utils.extract_above_threshold(SOURCE, TARGET, VALUE, threshold)

    # Not yet cythonized.
    nodes = [{
        'data': {'id': str(elem), 'label': str(str(elem) + ': ' + str(MZ[elem]))},
        'position': {'x':TSNE_DF["x"].iloc[elem], 
            'y':-TSNE_DF["y"].iloc[elem]},
        'classes':'None'} 
        for elem in range(0, TSNE_DF.shape[0])]
    #elements = nodes # initialize elements to just the nodes
    #print("Elements initialized using nodes: ", elements[0:10])
    ego = int(clust_selection)
    bdict = cython_utils.creating_branching_dict_new(s, t, ego, int(expand_level))
    edge_elems, edge_styles = cython_utils.generate_edge_elements_and_styles(bdict, s, t, nodes)
    #for key in bdict:
    #    print(key)
    #    print("NODES", bdict[key]["nodes"])
    #    print("EDGES", bdict[key]["edges"])

    elements = nodes + edge_elems
    base_node_style_sheet = [{
            'selector':'node', 'style':{'height':"100%", 'width':'100%', 
            'opacity':0.2, 'content':'data(label)', 'text-halign':'center',
            'text-valign':'center', "shape":"circle"}}]
    print(ego)
    ego_style = [
        {"selector":'node[id= "{}"]'.format(ego), 
        "style":{"shape":"diamond",'background-color':'gold',
            'opacity':0.95, 'height':'250%', 'width':'250%', 
            "border-color":"black", "border-width":10}}]
    print("DataStructure Generated, over to dash.cytoscape")
    #if True:
    #for element in elements[0:100] + elements[1110:]:
    #    print(element)
    if len(elements) <= 20000:
        out = html.Div([
            cyto.Cytoscape(
                id='cytoscape-tsne-subnet',
                layout={'name':'preset'},
                elements=elements,
                stylesheet= base_node_style_sheet + edge_styles + ego_style,
                boxSelectionEnabled=True,
                style={'width':'100%', 'height':'60vh', "border":"1px grey solid",
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
                style={'width':'100%', 'height':'60vh', "border":"1px grey solid",
                    "bg":"#feeff4"},
            ),
        ])
    return out



def generate_egonet(
    clust_selection, SM_MS2DEEPSCORE, TSNE_DF, threshold, expand_level):
    
    # Check input length, if not 1, 
    if not clust_selection:
        out = html.Div(html.H6(
            ("Please provide a node selection for EgoNet visualization.")))
        return html.Div(html.H6(["Please provide a node selection for fragmap."]))

    if len(clust_selection) > 1:
        print(("Warning: More than one node selected." +
            "Extracting first node in input as root for egonet." +
            f" Node = {clust_selection[0]}"))
    clust_selection = clust_selection[0]

    # Define Network Data
    n_nodes = SM_MS2DEEPSCORE.shape[0] # all nodes
    adj_m = _myfun.compute_adjacency_matrix(SM_MS2DEEPSCORE, threshold) # <-------------------- EXPENSIVE TO RUN EVERYTIME, FOR FULL DATASET; REQUIRED FOR DYNAMIC THRESHOLD AND EXPAND
    all_possible_edges = list(itertools.combinations(np.arange(0, n_nodes), 2))
    #edges = [{'data':{'id': str(elem[0]) + "-" + str(elem[1]), 
    #                    'source': str(elem[0]),
    #                    'target': str(elem[1])}} 
    #                    for elem in all_possible_edges if (adj_m[elem[0]][elem[1]] != 0)]
    
    edge_list = [
        (str(elem[0]), str(elem[1])) 
        for elem in all_possible_edges 
        if (adj_m[elem[0]][elem[1]] != 0)]                    
    edge_dict = create_edge_dict(edge_list)

    nodes = [{
        'data': {'id': str(elem), 'label': str('Node ' + str(elem))},
        'position': {'x':TSNE_DF["x"].iloc[elem], 
            'y':-TSNE_DF["y"].iloc[elem]}} 
        for elem in np.arange(0, n_nodes)]
    node_list = [str(elem) for elem in np.arange(0, 1267)]
    node_dict = create_node_dict(node_list, edge_dict)

    active_style_sheet = [{
        'selector':'node', 'style':{'height':"100%", 'width':'100%', 
        'opacity':0.8, 'content':'data(label)', 'text-halign':'center',
        'text-valign':'center', "shape":"circle"}}]
    
    
    elements = nodes # initialize elements to just the nodes
    ego = str(clust_selection)
    bdic = creating_branching_dict(ego, edge_dict, node_dict, 
        n_levels = expand_level)
    print(bdic)
    tmp_edge_elems, tmp_edge_styles = generate_edge_elements_and_styles(
        bdic, edge_dict)
    elements = elements + tmp_edge_elems
    active_style_sheet = active_style_sheet + tmp_edge_styles + [
        {"selector":'node[id= "{}"]'.format(ego), 
        "style":{"shape":"diamond",'background-color':'gold',
            'opacity':0.95, 'height':'250%', 'width':'250%', 
            "border-color":"black", "border-width":10}}]

    out = html.Div([
        cyto.Cytoscape(
            id='cytoscape-tsne-subnet',
            layout={'name':'preset'},
            elements= elements,
            stylesheet=active_style_sheet,
            boxSelectionEnabled=True,
            style={'width':'100%', 'height':'60vh', "border":"1px grey solid",
                "bg":"#feeff4"},
        ),
    ])
    return out

def create_edge_dict(edge_list):
    """Creates edge dict from edge list. 
    Edge list is a list of tuples [(Node1, Node2), (Node2, Node3), ...]. Nodes 
    must be strings or convertable to strings. Edges should exist only once:
    either N1 to N2 or N2 to N1. (unordered sets, arbitrary order in edge dict)
    """
    edge_dict = dict([( str(edge[0]) + "-" + str(edge[1]) , 
      (str(edge[0]), str(edge[1])) ) for edge in edge_list])
    return edge_dict

def create_node_dict(node_list, edge_dict):
    """Creates node dict from node_list and edge_dict objects."""
    node_dict = dict()
    for node in node_list:
        tmp = set()
        for edge_key in edge_dict:
            if node == edge_dict[edge_key][0] or node == edge_dict[edge_key][1]:
                tmp.add(edge_key)
        node_dict[str(node)] = tmp
    return node_dict

def creating_branching_dict(
    root_node, edge_dict, node_dict, n_levels, max_connections = 100):
    """Creates dictionary with branching connections from root.
    
    Root node is a string that uniquely identifies a node.
    n_levels indicates the depth of the branching.

    The branching dictionary keys returned are ordered from root to max level.
    """
    # Initialize data structures
    level_dict = {}
    all_nodes = set(root_node)
    all_edges = set(node_dict[root_node])
    level_dict["0"] = {"nodes":[root_node], "edges":node_dict[root_node]}
    for level in range(1, n_levels):
        tmp_edges = set()
        tmp_nodes = set()
        if len(level_dict[str(level-1)]["edges"]) == 0:
            print("No edges added at level:", level -1, 
                ". Aborting edge tracing at level:", level)
            break 
        # From edges of previous level construct new node list
        for edge in level_dict[str(level-1)]["edges"]:
            tmp_nodes.update(edge_dict[edge])
        tmp_nodes = tmp_nodes.difference(all_nodes)
        # Check whether any new nodes were connected
        if len(tmp_nodes) != 0:
            all_nodes.update(tmp_nodes)
            level_dict[str(level)] = {"nodes":tmp_nodes} # add level nodes
            # Find new edges, if any
            for node in tmp_nodes:
                # add all edges connecting to this node to tmp set
                tmp_edges.update(node_dict[node]) 
            tmp_edges = tmp_edges.difference(all_nodes) # get all new edges
        else:
            print("Stopping edge tracing at level:", level, 
                ". No new nodes were connected.")
            break
        if len(tmp_edges) != 0:
            all_edges.update(tmp_edges) # update all edges
            level_dict[str(level)]["edges"] = tmp_edges
        else:
            print("Stopping edge tracing at level:", level, 
                ". No new edges found.")
            break
        if len(all_nodes) >= max_connections:
            print( ("WARNING: exceeded max connections in local environment."
                "Number Nodes connected " f"{len(all_nodes)}" 
                "exceed max connections " f"{max_connections}"
                " Aborting edge tracing at level: " f"{level}"))
            break
    return(level_dict)



def generate_edge_elements_and_styles(branching_dict, edge_dict):
    """Generates an edge list and style list for a given branching dict. """
    n_colors = max(2, len(branching_dict))
    # "ice" might be a good alternative color
    colors = px.colors.sample_colorscale(
        "Viridis", [n/(n_colors -1) for n in range(n_colors)])
    opacities = np.arange(0.8, 0.1, step = -(0.8 - 0.1) / n_colors)
    widths = np.arange(5, 0.5, step = -(5 - 0.5) / n_colors)
    edge_elems = []
    edge_styles = []
    for idx, key in enumerate(branching_dict):
        print(idx, key, branching_dict[key])
        for edge in branching_dict[key]["edges"]:
            edge_elems.append({'data': {
                    'id':edge, 'source':edge_dict[edge][0],
                    'target':edge_dict[edge][1]}})
            edge_styles.append({
                "selector":'edge[id= "{}"]'.format(edge), 
                "style":{"line-color":colors[idx], 'opacity':opacities[idx], 
                'width':widths[idx]}}) #10 / max(1, idx)
        for node in branching_dict[key]["nodes"]:
            edge_styles.append({
                "selector":'node[id= "{}"]'.format(node), 
                "style":{"background-color":colors[idx], 
                'border-width':'2', 'border-color':'black', 
                'opacity':opacities[idx] }})
    return edge_elems, edge_styles