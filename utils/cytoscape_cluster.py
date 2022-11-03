import numpy as np
from utils import utils054
from utils import process_matchms as _myfun
import itertools
import dash_cytoscape as cyto
from dash import html
import plotly.graph_objects as go
from dash import dcc
def generate_cluster_node_link_diagram(TSNE_DF, clust_selection, SM_MS2DEEPSCORE, selected_class_data, color_dict):
    threshold = 0.75
    print("Cluster Selection:", clust_selection)
    
    # Define Network Data
    n_nodes = SM_MS2DEEPSCORE.shape[0] # all nodes
    adj_m = _myfun.compute_adjacency_matrix(SM_MS2DEEPSCORE, threshold) # <-------------------- EXPENSIVE TO RUN EVERYTIME, FOR FULL DATASET; REQUIRED FOR DYNAMIC THRESHOLD AND EXPAND
    all_possible_edges = list(itertools.combinations(np.arange(0, n_nodes), 2))
    edges = [{'data' : {'id': str(elem[0]) + "-" + str(elem[1]), 
                        'source': str(elem[0]),
                        'target': str(elem[1])}} 
                        for elem in all_possible_edges if (adj_m[elem[0]][elem[1]] != 0)]
    
    nodes = [{'data': {'id': str(elem)}, 'label': 'Node ' + str(elem),
        'position': {'x': TSNE_DF["x"].iloc[elem], 'y': -TSNE_DF["y"].iloc[elem]}, 
        'classes': selected_class_data[elem]} # <-- Added -y_coord for correct orientation in line with t-sne
        for elem in np.arange(0, n_nodes)]

    print(len(nodes))
    active_style_sheet = [{'selector' : 'edge', 'style' : {'opacity' : 0.4}}, 
                            {'selector' : 'node', 'style' : {'height' : "100%", 'width' : '100%', 'opacity' : 0.60}}]
    clust_selection = [str(elem) for elem in clust_selection]
    print(clust_selection)
    my_set = set(clust_selection)
    print(my_set)
    sel_edges = []
    sel_nodes = []
    sel_classes = set()
    
    
    
    
    for elem in edges:
        if (elem["data"]["source"] in my_set) and (elem["data"]["target"] in my_set):
            sel_edges.append(elem)
    for elem in nodes:
        if elem["data"]["id"] in my_set:
            sel_nodes.append(elem)
            sel_classes.add(elem["classes"])
    style_sheet = active_style_sheet + [{'selector': f".{clust}",
    'style': { 'background-color': f"{color_dict[clust]}"}} for clust in list(sel_classes)]
    print(sel_nodes)
    
    if len(sel_edges) + len(sel_nodes) <= 10000:
        out = html.Div([
            cyto.Cytoscape(
                id='cytoscape-tsne-subnet',
                layout={'name': 'preset'},
                elements= sel_edges + sel_nodes,
                stylesheet=style_sheet,
                boxSelectionEnabled=True,
                style={'width': '100%', 'height': '60vh', "border":"1px grey solid", "bg" : "#feeff4"},
            ),
        ])
        return out
    else:
        edge_x = []
        edge_y = []


        for edge in sel_edges:
            # node id's are integer locations
            x0 = TSNE_DF["x"].iloc[int(edge["data"]["source"])]
            y0 = TSNE_DF["y"].iloc[int(edge["data"]["source"])]
            x1 = TSNE_DF["x"].iloc[int(edge["data"]["target"])]
            y1 = TSNE_DF["y"].iloc[int(edge["data"]["target"])]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(40, 40, 40, 0.2)'),
            hoverinfo='none',
            mode='lines')
        node_x = []
        node_y = []
        for node in nodes:
            x =  node["position"]["x"]
            y = -node["position"]["y"] # inverting back...
            node_x.append(x)
            node_y.append(y)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color ='steelblue',
                size=4,
                line_width=1))
        fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Static Cluster NLD', title_x=0.01, title_y=0.01,
                    titlefont_size = 12,
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                    )
        fig.update_layout(clickmode='event+select', 
                    margin = {"autoexpand":True, "b" : 0, "l":0, 
                            "r":0, "t":0})
        out = html.Div([dcc.Graph(id = "cluster_express", figure=fig, 
            style={"width":"100%","height":"60vh", "border":"1px grey solid"}, 
            config={'staticPlot': True}
            )])
        return out