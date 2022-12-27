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
from specxplore import dashboard_components as _dc
from specxplore import loading as load_utils
from specxplore import visuals as visual_utils
from specxplore import process_matchms as _myfun
from specxplore import egonet
from specxplore import augmap
from specxplore import tsne_plotting
from specxplore import cytoscape_cluster
from specxplore import fragmap
from specxplore import parsing
import cython_utils
import pickle
#import seaborn as sns
import copy
import itertools
import dash_cytoscape as cyto
import plotly.graph_objects as go
from scipy.cluster import hierarchy

app=Dash(__name__)

# TODO: load necessary data.
# --> pandas df with spec_id, x and y coord, and classification columns.
# --> spectrum list with list index corresponding to spec_id (for now)
# TODO: load necessary data.
# --> pandas df with spec_id, x and y coord, and classification columns.
# --> spectrum list with list index corresponding to spec_id (for now)
global STRUCTURE_DICT
global CLASS_DICT
#STRUCTURE_DICT, CLASS_DICT=load_utils.process_structure_class_table(
#    "data/classification_table.csv")
CLASS_DICT=load_utils.extract_classes_from_ms2query_results(
    "data-input/results/GNPS-NIH-NATURALPRODUCTSLIBRARY.csv")
global AVAILABLE_CLASSES
AVAILABLE_CLASSES=list(CLASS_DICT.keys())
#print(AVAILABLE_CLASSES)


global SM_MS2DEEPSCORE
global SM_MODIFIED_COSINE
global SM_SPEC2VEC
SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC=load_utils.load_pairwise_sim_matrices()

global TSNE_DF
with open("data/tsne_df.pickle", 'rb') as handle:
    TSNE_DF=pickle.load(handle)

# Initializing color dict
selected_class_data=CLASS_DICT[AVAILABLE_CLASSES[0]]
# Create overall figure with color_dict mapping
n_colors=len(set(selected_class_data)) # TODO: speed this up using n_clust argument that is pre-computed
colors=visual_utils.construct_grey_palette(n_colors, white_buffer=20)
init_color_dict=visual_utils.create_color_dict(colors, selected_class_data)

global ALL_SPEC_IDS
ALL_SPEC_IDS=TSNE_DF.index # <-- add list(np.unique(spec_id_list of sorts))

global ALL_SPECTRA
file=open("data/cleaned_demo_data.pickle", 'rb')
ALL_SPECTRA=pickle.load(file)
file.close()

# PROTOTYPING CYTHON DATA STRUCTURES
SOURCE, TARGET, VALUE = cython_utils.construct_long_format_sim_table(SM_MS2DEEPSCORE)
print(TSNE_DF)
NODE_ID = np.array(TSNE_DF["id"], dtype=np.int32)
NODE_X = np.array(TSNE_DF["x"], dtype=np.double)
NODE_Y = np.array(TSNE_DF["y"], dtype=np.double)

file = open("data/extracted_precursor_mz_values.pickle", 'rb')
MZ = pickle.load(file) 
file.close()

app=dash.Dash(external_stylesheets=[dbc.themes.YETI]) # MORPH or YETI style.
app.layout=html.Div([
    dbc.Row([
        dbc.Col([html.H1([html.B("specXplore prototype")], 
            style={"margin-bottom": "-0.1em"})], width=6)]),
    dbc.Row([
        dbc.Col(
            [html.H6("Authors: Kevin Mildau - Henry Ehlers")], width=7),
        dbc.Col(
            dcc.Tabs(id="right-panel-tab-group", value='right-panel-tab', 
                children=[
                    dcc.Tab(label='Cluster', value='tab-cluster'),
                    dcc.Tab(label='EgoNet', value='tab-egonet'),
                    dcc.Tab(label='Augmap', value='tab-augmap'),
                    dcc.Tab(label='Settings', value='tab-settings'),
                    dcc.Tab(label='Data View', value='tab-data')]), 
                width=5)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([dcc.Graph(id="tsne-overview-graph", figure={}, 
            style={"width":"100%","height":"60vh", 
            "border":"1px grey solid"})], width=7),
        dbc.Col([html.Div(id='right-panel-tabs-content')], width=5),
    ], style={"margin-bottom": "-1em"}),
    html.Br(),
    dbc.Row([
        dbc.Col([html.H6("Selected Points for Cluster View:")], width=6),
        dbc.Col([html.H6("Set Edge Threshold:")], width=2),
        dbc.Col([dcc.Input( id="threshold_text_input", type="number", 
            debounce=True, placeholder="Threshold 0 < thr < 1, def. 0.9", 
            style={"width" : "100%"})], width=4)]),
    dbc.Row([
        dbc.Col([dcc.Dropdown(id='clust-dropdown', multi=True, 
            style={'width': '100%', 'font-size': "75%"}, 
            options=ALL_SPEC_IDS)], width=6),
        dbc.Col([html.H6("Selected Points for Focus View:")], width=2),
        dbc.Col([dcc.Dropdown(id='focus_dropdown', multi=True, 
            style={'width': '100%', 'font-size': "75%"}, 
            options=ALL_SPEC_IDS)], width=4)]),
    dbc.Row([
        dbc.Col([html.Div( style={'width': '100%'})], width=6),
        dbc.Col([html.H6("Reload open tab:")],width=4),
        dbc.Col([dbc.Button('Submit Reload', id='refresh-open-tab-button', 
            style={"width":"100%"})], width=2),]),
    dbc.Row([
        dbc.Col([html.Div( style={'width': '100%'})], width=6),
        dbc.Col([html.H6("Set expand level:")], width=4),
        dbc.Col([dcc.Input( id="expand_level_input", type="number", 
            debounce=True, 
            placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
            style={"width" : "100%"})], width=2)]),
    html.Br(),
    dbc.Row([
        dbc.Col([dbc.Button("Generate Fragmap", id="push_fragmap", 
            style={"width":"100%"})], width=2),
        dbc.Col([dbc.Button("Generate Spectrum Plot", 
            style={"width": "100%"})], width=2),
        dbc.Col([dbc.Button("Show Spectrum Data", style={"width": "100%"})],
            width=2),
        dbc.Col([dcc.Dropdown(id='class-dropdown' , multi=False, 
            clearable=False, options=AVAILABLE_CLASSES, 
            value=AVAILABLE_CLASSES[5])], width=4),
    dbc.Col([dbc.Button("Push Class Selection", id="push-class", 
        style={"width":"100%"})], width=2)]),
    dcc.Store(id="edge_threshold", data=0.9),
    dcc.Store(id="expand_level", data=int(1)),
    dcc.Store(id="selected_class_level", data=AVAILABLE_CLASSES[0]),
    dcc.Store(id="selected_class_data", data=CLASS_DICT[AVAILABLE_CLASSES[0]]),
    dcc.Store(id="color_dict",  data=init_color_dict),
    html.Br(),
    dbc.Row([dbc.Col([html.Div(id="fragmap_panel", 
        style={"width":"100%", "border":"1px grey solid"})], width=12)]),
    html.Br(),
    dbc.Row([dbc.Col([html.Div(id="data-panel", 
        style={"width":"100%", "border":"1px grey solid"})], width=12)])], 
        style={"width" : "100%"})

@app.callback([Output("edge_threshold", "data"),
               Output("threshold_text_input", "placeholder")],
              [Input('threshold_text_input', 'n_submit'),
              Input("threshold_text_input", "value")])

def update_threshold_trigger_handler(n_submit, new_threshold):
    new_threshold, new_placeholder=parsing.update_threshold(new_threshold)
    return new_threshold, new_placeholder

@app.callback(
    Output("expand_level", "data"),
    Output("expand_level_input", "placeholder"),
    Input('expand_level_input', 'n_submit'),
    Input("expand_level_input", "value"))

def expand_trigger_handler(n_submit, new_expand_level):
    new_expand_level, new_placeholder=parsing.update_expand_level(
        new_expand_level)
    return new_expand_level, new_placeholder

# GLOBAL OVERVIEW UPDATE TRIGGER
@app.callback(
    Output("tsne-overview-graph", "figure"), 
    Input("push-class", "n_clicks"),
    Input("tsne-overview-graph", "clickData"), # or "hoverData"
    Input("selected_class_level", "data"),
    Input("selected_class_data", "data"),
    State("color_dict", "data"))

def left_panel_trigger_handler(
    n_clicks, 
    point_selection, 
    selected_class_level, 
    selected_class_data, 
    color_dict):
    """ Modifies global overview plot in left panel """
    tsne_fig=tsne_plotting.plot_tsne_overview(
        point_selection, selected_class_level, selected_class_data, TSNE_DF, 
        color_dict)
    return tsne_fig

# CLASS SELECTION UPDATE ------------------------------------------------------
@app.callback(
    Output("selected_class_level", "data"), 
    Output("selected_class_data", "data"),
    Output("color_dict", "data"),   
    Input("push-class", "n_clicks"),
    State("class-dropdown", "value"))
def class_update_trigger_handler(n_clicks, selected_class):
    """ Wrapper Function that construct class dcc.store data. """
    selected_class_data, color_dict=parsing.update_class(selected_class, 
        CLASS_DICT)
    return selected_class, selected_class_data, color_dict

# RIGHT PANEL BUTTON CLICK UPDATES --------------------------------------------
@app.callback(
    Output('right-panel-tabs-content', 'children'),
    Input('right-panel-tab-group', 'value'),
    Input('refresh-open-tab-button', 'n_clicks'), # trigger only
    State('clust-dropdown', 'value'), 
    State("color_dict", "data"),
    State("selected_class_data", "data"),
    State("edge_threshold", "data"),
    State("expand_level", "data"))
def right_panel_trigger_handler(
    tab, n_clicks, clust_selection, color_dict, selected_class_data, threshold, 
    expand_level):
    if tab == "tab-cluster" and clust_selection:
        panel = cytoscape_cluster.generate_cluster_node_link_diagram_cythonized(
            TSNE_DF, clust_selection, SM_MS2DEEPSCORE, selected_class_data,
            color_dict, threshold, SOURCE, TARGET, VALUE, MZ)
        #panel=cytoscape_cluster.generate_cluster_node_link_diagram(
        #    TSNE_DF, clust_selection, SM_MS2DEEPSCORE, selected_class_data, 
        #    color_dict, threshold)
    elif tab == "tab-egonet"  and clust_selection:
        panel = egonet.generate_egonet_cythonized(clust_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, threshold, expand_level)
        #panel=egonet.generate_egonet(
        #    clust_selection, SM_MS2DEEPSCORE, TSNE_DF, threshold, expand_level)
    elif tab == "tab-augmap"  and clust_selection:
        panel=augmap.generate_augmap(
            clust_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, 
            threshold)
    elif tab == "tab-settings":
        panel=[html.H6("Settings panel inclusion pending.")]
    elif tab == "tab-data":
        panel=[html.H6("Data panel inclusion pending.")]
    else:
        warning("Nothing selected for display in right panel yet.")
        panel=[html.H6("empty-right-panel")]
    return panel

# tsne-overview selection data trigger ----------------------------------------
@app.callback(
    Output('clust-dropdown', 'value'),
    Input('tsne-overview-graph', 'selectedData'))
def plotly_selected_data_trigger(plotly_selection_data):
    """ Wrapper Function for tsne point selection handling. """
    selected_ids=parsing.extract_identifiers(plotly_selection_data)
    return selected_ids

# Fragmap trigger -------------------------------------------------------------
@app.callback(
    Output('fragmap_panel', 'children'),
    Input('push_fragmap', 'n_clicks'), # trigger only
    State('clust-dropdown', 'value'))
def fragmap_trigger(n_clicks, selection_data):
    """ Wrapper function that calls fragmap generation modules. """
    # Uses: global variable ALL_SPECTRA
    fragmap_panel=fragmap.generate_fragmap_panel(selection_data, ALL_SPECTRA)
    return fragmap_panel

if __name__ == '__main__':
    app.run_server(debug=True)
