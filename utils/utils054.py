import plotly.express as px
import numpy as np

def create_edge_dict(edge_list):
    """Creates edge dict from edge list. 
    Edge list is a list of tuples [(Node1, Node2), (Node2, Node3), ...]. Nodes must
    be strings or convertable to strings. Edges should exist only once: 
    either N1 to N2 or N2 to N1. (unordered sets, arbitrary order in edge dict.)
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

def creating_branching_dict(root_node, edge_dict, node_dict, n_levels):
    """Creates dictionary with branching connections from root.
    
    Root node is a string that uniquely identifies a node.
    n_levels indicates the depth of the branching.

    The branching dictionary keys returned are ordered from root to max level.
    """
    # Initialize data structures
    level_dict = {}
    all_nodes = set(root_node)
    all_edges = set(node_dict[root_node])
    level_dict["0"] = {"nodes" : [root_node], "edges" : node_dict[root_node]}
    for level in range(1, n_levels):
        tmp_edges = set()
        tmp_nodes = set()
        if len(level_dict[str(level-1)]["edges"]) == 0:
            print("No edges added at level:", level -1, ". Aborting edge tracing at level: ", level)
            break 
        # From edges of previous level construct new node list
        for edge in level_dict[str(level-1)]["edges"]:
            tmp_nodes.update(edge_dict[edge])
        tmp_nodes = tmp_nodes.difference(all_nodes)
        if len(tmp_nodes) != 0: # Check whether any new nodes were connected
            all_nodes.update(tmp_nodes)
            level_dict[str(level)] = {"nodes" : tmp_nodes} # add level nodes
            # Find new edges, if any
            for node in tmp_nodes:
                tmp_edges.update(node_dict[node]) # add all edges connecting to this node to tmp set
            tmp_edges = tmp_edges.difference(all_nodes) # get all new edges
        else:
            print("Stopping edge tracing at level: ", level, ". No new nodes were connected.")
            break
        if len(tmp_edges) != 0:
            all_edges.update(tmp_edges) # update all edges
            level_dict[str(level)]["edges"] = tmp_edges
        else:
            print("Stopping edge tracing at level: ", level, ". No new edges found.")
            break
    
    return(level_dict)



def generate_edge_elements_and_styles(branching_dict, edge_dict):
    """Generates an edge list and style list for a given branching dict. """
    n_colors = max(2, len(branching_dict))
    colors = px.colors.sample_colorscale("Viridis", [n/(n_colors -1) for n in range(n_colors)]) # "ice" might also be a good option here.
    opacities = np.arange(0.8, 0.1, step = -(0.8 - 0.1) / n_colors)
    widths = np.arange(5, 0.5, step = -(5 - 0.5) / n_colors)
    edge_elems = []
    edge_styles = []
    for idx, key in enumerate(branching_dict):
        print(idx, key, branching_dict[key])
        for edge in branching_dict[key]["edges"]:
            edge_elems.append({'data' : {'id': edge, 'source': edge_dict[edge][0],'target': edge_dict[edge][1]}})
            edge_styles.append({"selector": 'edge[id= "{}"]'.format(edge), "style": {"line-color": colors[idx], 'opacity': opacities[idx], 'width': widths[idx]}}) #10 / max(1, idx)
        for node in branching_dict[key]["nodes"]:
            edge_styles.append({"selector": 'node[id= "{}"]'.format(node), "style": {"background-color": colors[idx], 
              'border-width':'2', 'border-color':'black', 'opacity': opacities[idx] }})
    return edge_elems, edge_styles