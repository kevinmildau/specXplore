# Main specXplore prototype
from logging import warning
import dash
from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from utils import dashboard_components as _dc
import pickle
import seaborn as sns
import copy
# import kmedoid


app = Dash(__name__)

# TODO: load necessary data.
# --> pandas df with spec_id, x and y coord, and classification columns.
# --> spectrum list with list index corresponding to spec_id (for now)
# TODO: load necessary data.
# --> pandas df with spec_id, x and y coord, and classification columns.
# --> spectrum list with list index corresponding to spec_id (for now)

# TODO: load pairwise similarity matrices. (for now)
# --> They should be passed on to any functions creating edge lists, as well as to the heatmap
# TODO: "Load" pairwise similarity matrices using numpy on disk indexing (need to be saved accordingly)

# TODO: Add ego net construction function (simple, no expansion)
# TODO: Add cluster net construction function (simple, no expansion)

# TODO: Add style modifier for direct connections (edges)

# TODO: Add grayscale cluster coloring. 
# TODO: Add highlight cluster in neon-pink upon hover. (#FF10F0)

# TODO: add toggle for different clustering styles. Allow for different chemical classes, or different k-medoid settings. 
# Extract those settings from a pre-computed pandas df.


# TODO: Add dropdown intermediate structure to contain intermediate spec_id list. Should be modifiable (additions and removals).
# TODO: Add dropdown intermediate structure to contain focus spec_id list. Should be modifiable (additions and removals).



# TODO: add t-sne load funciton & make actual t-sne plot
df = px.data.iris() # iris is a pandas DataFrame
df["id"] = df.index # + 1 for indexing start at one
fig = px.scatter(df, x="sepal_width", y="sepal_length", custom_data=["id"], hover_data=["id"])
fig.update_layout(clickmode='event+select', margin = {"autoexpand":True, "b" : 0, "l":0, "r":0, "t":0})
fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, 
  xaxis_visible=False, xaxis_showticklabels=False, title_text='T-SNE Overview', title_x=0.99, title_y=0.01)

# TODO: Construct global static variable to contain all possible spectrum identifiers
global ALL_SPEC_IDS
ALL_SPEC_IDS = df.index # <-- add list(np.unique(spec_id_list of sorts))

# TODO: load pairwise similarity matrices. (for now)
# --> They should be passed on to any functions creating edge lists, as well as to the heatmap
# TODO: "Load" pairwise similarity matrices using numpy on disk indexing (need to be saved accordingly)

# TODO: Add ego net construction function (simple, no expansion)
# TODO: Add cluster net construction function (simple, no expansion)

# TODO: Add style modifier for direct connections (edges)

# TODO: Add grayscale cluster coloring. 
# TODO: Add highlight cluster in neon-pink upon hover. (#FF10F0)

# TODO: add toggle for different clustering styles. Allow for different chemical classes, or different k-medoid settings. 
# Extract those settings from a pre-computed pandas df.


# TODO: Add dropdown intermediate structure to contain intermediate spec_id list. Should be modifiable (additions and removals).
# TODO: Add dropdown intermediate structure to contain focus spec_id list. Should be modifiable (additions and removals).

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
        dbc.Col([dcc.Graph(id="tsne-overview-graph", figure=fig, style={"width":"100%","height":"60vh", "border":"1px grey solid"})], 
            width=6),
        dbc.Col([html.Div(id = "right-panel-view")], 
            width=6),
    ], style = {"margin-bottom": "-1em"}),
    html.Br(),
    # Additional controls, dropdown & 
    dbc.Row([
        dbc.Col([html.H6("Selected Points for Cluster View:")], width = 6),
        dbc.Col([html.H6("Selected Points for Focus View:")],width=6)
    ]),
    dbc.Row([
        dbc.Col([dcc.Dropdown(id='clust-dropdown' , multi=True)], width = 4),
        dbc.Col([dbc.Button("Push Cluster Selection!",style={"width":"100%"})],width=2), 
        dbc.Col([dcc.Dropdown(id='focus-dropdown' , multi=True)], width = 4),
        dbc.Col([dbc.Button("Push Focus Selection!",style={"width":"100%"})],width=2), 
    ]),
    html.Br(),
    dbc.Row([
    dbc.Col([dbc.Button("Generate Fragmap",style={"width":"100%"})], width=2),
    dbc.Col([dbc.Button("Generate Spectrum Plot", style={"width": "100%"})],width=2),
    dbc.Col([dbc.Button("Show Spectrum Data", style={"width": "100%"})],width=2)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([html.Div(id = "focus-panel-view-1", style={"width":"100%","height":"60vh", "border":"1px grey solid"})], width=6),
        dbc.Col([html.Div(id = "focus-panel-view-2", style={"width":"100%","height":"60vh", "border":"1px grey solid"})], width=3),
        dbc.Col([html.Div(id = "focus-panel-view-3", style={"width":"100%","height":"60vh", "border":"1px grey solid"})], width=3),]
    )
], style = {"width" : "100%"})

@app.callback(Output('right-panel-view', 'children'),
              [Input('toggle-clust', 'n_clicks'),
              Input('toggle-ego', 'n_clicks'),
              Input('toggle-aug', 'n_clicks'),
              Input('toggle-set', 'n_clicks'),
              Input('toggle-data', 'n_clicks')],
              State('clust-dropdown', 'value'))
def update_output_clust(n_clicks_clust, n_clicks_ego, n_clicks_aug, n_clicks_set, n_clicks_data, clust_selection = None):
    if "toggle-clust" == ctx.triggered_id:
        print("Clust clicked.")
        tmp = copy.deepcopy(df)
        tmp = tmp.iloc[clust_selection]
        subfig = px.scatter(tmp, x="sepal_width", y="sepal_length", custom_data=["id"], hover_data=["id"])
        subfig.update_traces(marker=dict(size=25))
        subfig.update_layout(clickmode='event+select', margin = {"autoexpand":True, "b" : 0, "l":0, "r":0, "t":0})
        subfig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, 
            xaxis_visible=False, xaxis_showticklabels=False, title_text='T-SNE Overview', title_x=0.99, title_y=0.01)
        subfig.update_layout(title_text = "A totally new clust view!")
        out = [dcc.Graph(id="cluster-graph", figure=subfig, style={"width":"100%", "height":"60vh", "border":"1px grey solid"})]
        return out
    if "toggle-ego" == ctx.triggered_id:
        print("Ego clicked.")
        dummy = copy.deepcopy(fig) # fig is global defined
        dummy.update_layout(title_text = "A totally new ego view!")
        out = [dcc.Graph(id="ego-graph", figure=dummy, style={"width":"100%", "height":"60vh", "border":"1px grey solid"})]
        return out
    if "toggle-aug" == ctx.triggered_id:
        out = [html.H6("Heatmap inclusion pending.")]
        return out
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
