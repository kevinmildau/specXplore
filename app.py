# Main specXplore prototype
from logging import warning
import dash
from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly
import pandas as pd
import numpy as np
from utils import dashboard_components as _dc
from utils import loading as load_utils
from utils import visuals as visual_utils
from utils import process_matchms as _myfun
from utils import utils054
from utils import egonet
from utils import augmap
from utils import tsne_plotting
from utils import cytoscape_cluster
import pickle
import seaborn as sns
import copy
import itertools
import dash_cytoscape as cyto
import plotly.graph_objects as go
from scipy.cluster import hierarchy

app = Dash(__name__)

# TODO: load necessary data.
# --> pandas df with spec_id, x and y coord, and classification columns.
# --> spectrum list with list index corresponding to spec_id (for now)
# TODO: load necessary data.
# --> pandas df with spec_id, x and y coord, and classification columns.
# --> spectrum list with list index corresponding to spec_id (for now)
global STRUCTURE_DICT
global CLASS_DICT
STRUCTURE_DICT, CLASS_DICT = load_utils.process_structure_class_table("data/classification_table.csv")
global AVAILABLE_CLASSES
AVAILABLE_CLASSES = list(CLASS_DICT.keys())
#print(AVAILABLE_CLASSES)

# TODO: [ ] "Load" pairwise similarity matrices using numpy on disk indexing (need to be saved accordingly)
# TODO: [ ] Create more dynamic data structures to allow between 1 and 5 similarity scores. 
#       [ ] implement corresponding layering in heatmap visual and hover inf.
global SM_MS2DEEPSCORE
global SM_MODIFIED_COSINE
global SM_SPEC2VEC
SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC = load_utils.load_pairwise_sim_matrices()

global TSNE_DF
with open("data/tsne_df.pickle", 'rb') as handle:
    TSNE_DF = pickle.load(handle)

# Initializing color dict
selected_class_data = CLASS_DICT[AVAILABLE_CLASSES[0]]
# Create overall figure with color_dict mapping
n_colors = len(set(selected_class_data)) # TODO: speed this up using n_clust argument that is pre-computed
colors = visual_utils.construct_grey_palette(n_colors, white_buffer = 20)
init_color_dict = visual_utils.create_color_dict(colors, selected_class_data)

global ALL_SPEC_IDS
ALL_SPEC_IDS = TSNE_DF.index # <-- add list(np.unique(spec_id_list of sorts))


app = dash.Dash(external_stylesheets=[dbc.themes.YETI]) # MORPH or YETI style.
app.layout = \
html.Div\
([
    # Title
    dbc.Row([dbc.Col([html.H1([html.B("specXplore prototype")], style = {"margin-bottom": "-0.1em"})], width=6)]),
    # Subtitle & mid view selectors
    dbc.Row([
        dbc.Col([ 
            html.H6(html.H6("Authors: Kevin Mildau - Henry Ehlers"),)], 
            width=7
        ),
        dbc.Col([dbc.Button("Cluster",   id = "toggle-clust", style={"width":"100%", "font-size" : 12}, size="sm")], width=1),
        dbc.Col([dbc.Button("EgoNet",    id = "toggle-ego",   style={"width":"100%", "font-size" : 12}, size="sm")], width=1),
        dbc.Col([dbc.Button("Augmap",    id = "toggle-aug",   style={"width":"100%", "font-size" : 12}, size="sm")], width=1),
        dbc.Col([dbc.Button("Settings",  id = "toggle-set",   style={"width":"100%", "font-size" : 12}, size="sm")], width=1),
        dbc.Col([dbc.Button("Data View", id = "toggle-data",  style={"width":"100%", "font-size" : 12}, size="sm")], width=1),
    ], style = {"margin-bottom": "-1em"}),
    html.Br(),
    # Plots
    dbc.Row([
        dbc.Col([dcc.Graph(id = "tsne-overview-graph", figure={}, style={"width":"100%","height":"60vh", "border":"1px grey solid"})], 
            width=7),
        dbc.Col([html.Div(id = "right-panel-view")], 
            width=5),
    ], style = {"margin-bottom": "-1em"}),
    html.Br(),
    # Additional controls, dropdown & 
    dbc.Row([
        dbc.Col([html.H6("Selected Points for Cluster View:")], width = 6),

        # -->
        dbc.Col([html.H6("Set Edge Threshold:")], width = 2),
        dbc.Col([dcc.Input(
            id="threshold_text_input",
            type="number", debounce = True,
            placeholder = "Threshold 0 < thr < 1, def. 0.9", style = {"width" : "100%"}
        )], width = 4),

        
    ]),
    dbc.Row([
        dbc.Col([dcc.Dropdown(id='clust-dropdown' , multi=True, style={'width': '100%', 'font-size': "75%"})], 
                              width = 6),

        # -->
        dbc.Col([html.H6("Selected Points for Focus View:")],width=2),

        dbc.Col([dcc.Dropdown(id='focus-dropdown' , multi=True)], width = 4),
    ]),
    html.Br(),
    dbc.Row([
    dbc.Col([dbc.Button("Generate Fragmap",style={"width":"100%"})], width=2),
    dbc.Col([dbc.Button("Generate Spectrum Plot", style={"width": "100%"})],width=2),
    dbc.Col([dbc.Button("Show Spectrum Data", style={"width": "100%"})],width=2),
    dbc.Col([dcc.Dropdown(id='class-dropdown' , multi=False, clearable=False, options = AVAILABLE_CLASSES, value = AVAILABLE_CLASSES[0])], width = 4),
    dbc.Col([dbc.Button("Push Class Selection", id = "push-class",style={"width":"100%"})],width=2)
    ]),
    dcc.Store(id = "edge_threshold", data = 0.9),
    dcc.Store(id = "selected_class_level", data = AVAILABLE_CLASSES[0]),
    dcc.Store(id = "selected_class_data",  data = CLASS_DICT[AVAILABLE_CLASSES[0]]),
    dcc.Store(id = "color_dict",  data = init_color_dict),
    html.Br(),
    dbc.Row([
        dbc.Col([html.Div(id = "focus-panel-view-1", style={"width":"100%", "border":"1px grey solid"})], width=6),
        dbc.Col([html.Div(id = "focus-panel-view-2", style={"width":"100%", "border":"1px grey solid"})], width=6)]
    )
], style = {"width" : "100%"})


@app.callback([Output("edge_threshold", "data"),
               Output("threshold_text_input", "placeholder")],
              [Input('threshold_text_input', 'n_submit'),
              Input("threshold_text_input", "value")])

def update_threshold(n_submit, new_threshold):
    print("Text Input Threshold triggered")
    print(type(new_threshold), new_threshold)
    default_threshold = 0.9
    default_placeholder = "Threshold 0 < thr < 1, def. 0.9"
    if new_threshold:
        print("Passed ")
        if new_threshold < 1 and new_threshold > 0:
            print("That's a good threshold!")
            placeholder = f'Threshold 0 < thr < 1, current: {new_threshold}'
            return new_threshold,  placeholder
    print("Bad threshold input. Restoring defaults.")
    return default_threshold,  default_placeholder

# UPDATE GLOBAL OVERVIEW VIA CLASS SELECTION
@app.callback(Output("tsne-overview-graph", "figure"), 
              [Input("push-class", "n_clicks"),
              Input("tsne-overview-graph", "clickData"),
              Input("class-dropdown", "value"),
              Input("selected_class_data", "data")])

def update_tsne_overview(n_clicks, hoverData, selected_class_level, selected_class_data):
    """ Modifies global overview plot """
    
    # Create overall figure with color_dict mapping
    n_colors = len(set(selected_class_data)) # TODO: speed this up using n_clust argument that is pre-computed
    colors = visual_utils.construct_grey_palette(n_colors, white_buffer = 20)
    color_dict = visual_utils.create_color_dict(colors, selected_class_data)
    
    if hoverData:  
        # Modify class color for the class of the hovered over point.
        selected_point = hoverData["points"][0]["customdata"][0]
        color_dict[selected_class_data[selected_point]] = "#FF10F0"
    
    return tsne_plotting.plot_tsne_overview(TSNE_DF, color_dict, 
                                            selected_class_data, 
                                            selected_class_level)

# UPDATED CLASS SELECTION SAVE IN DCC STORE
@app.callback(Output("selected_class_level", "data"), 
              Output("selected_class_data", "data"),
              Output("color_dict", "data"),   
              Input("push-class", "n_clicks"),
              State("class-dropdown", "value"))
def update_class_selection(n_clicks, value):
    selected_class_data = CLASS_DICT[value]
    # Create overall figure with color_dict mapping
    n_colors = len(set(selected_class_data)) # TODO: speed this up using n_clust argument that is pre-computed
    colors = visual_utils.construct_grey_palette(n_colors, white_buffer = 20)
    color_dict = visual_utils.create_color_dict(colors, selected_class_data)
    return value, selected_class_data, color_dict

# RIGHT PANEL BUTTON CLICK UPDATES
@app.callback(Output('right-panel-view', 'children'),
              [Input('toggle-clust', 'n_clicks'),
              Input('toggle-ego', 'n_clicks'),
              Input('toggle-aug', 'n_clicks'),
              Input('toggle-set', 'n_clicks'),
              Input('toggle-data', 'n_clicks')],
              State('clust-dropdown', 'value'), 
              State("color_dict", "data"),
              State("selected_class_data", "data"),
              State("edge_threshold", "data"))
def update_output_clust(n_clicks_clust, n_clicks_ego, n_clicks_aug, n_clicks_set, n_clicks_data, clust_selection, 
                        color_dict, selected_class_data, threshold):
    if "toggle-clust" == ctx.triggered_id and clust_selection:
        print("In Clust Selection")
        return cytoscape_cluster.generate_cluster_node_link_diagram(TSNE_DF, clust_selection, SM_MS2DEEPSCORE, selected_class_data, color_dict, threshold)
    if "toggle-ego" == ctx.triggered_id and clust_selection:
        return egonet.generate_egonet(clust_selection, SM_MS2DEEPSCORE, TSNE_DF, threshold)
    if "toggle-aug" == ctx.triggered_id and clust_selection:
        return augmap.generate_augmap(clust_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, threshold)
    if "toggle-set" == ctx.triggered_id:
        out = [html.H6("Settings panel inclusion pending.")]
        return out
    if "toggle-data" == ctx.triggered_id:
        out = [html.H6("Data panel inclusion pending.")]
        return out
    else:
        warning("Nothing selected for display in right panel yet.")
        out = [html.H6("empty-right-panel")]
        return out


# PLOTLY GLOBAL OVERVIEW POINT SELECTION PUSH TO DROPDOWN
@app.callback(
    [Output('clust-dropdown', 'options'),
     Output('clust-dropdown', 'value')],
    Input('tsne-overview-graph', 'selectedData'))
def extract_identifiers(plotly_selection_data):
    """ Function extracts custom_data id's from a provided point selection dictionary."""
    # Edge Cases
    # startup empty selection
    # no selection data
    # broken custom_data or re-arranged columns --> change to col identifier use
    print("Triggered clust-dropdown input function.")
    if plotly_selection_data != None:
        selected_ids = [elem["customdata"][0] for elem in plotly_selection_data["points"]]
        print("slected _ids:", selected_ids)
    else:
        selected_ids = []
    return ALL_SPEC_IDS, selected_ids # all_spec_id is constant, global

if __name__ == '__main__':
    app.run_server(debug=True)
