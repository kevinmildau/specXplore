# Main specXplore prototype
from dash import dcc, html, ctx, dash_table, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from specxplore import egonet, augmap, tsne_plotting, clustnet, fragmap, data_transfer, specxplore_data_cython, specxplore_data
import pickle
import plotly.graph_objects as go
import os

# Load specxplore data object from file
specxplore_input_file = os.path.join("data_import_testing", "results", "phophe_specxplore.pickle")
with open(specxplore_input_file, 'rb') as handle:
    GLOBAL_DATA = pickle.load(handle) 

# Define global variables
global CLASS_DICT # Dictionary with key for each possible classification scheme, and list with class assignments
global AVAILABLE_CLASSES # List of class strings corresponding to keys in class_dict
global SM_MS2DEEPSCORE # ndarray - pairwise similarity matrix
global SM_MODIFIED_COSINE # ndarray - pairwise similarity matrix
global SM_SPEC2VEC # ndarray - pairwise similarity matrix
global TSNE_DF # pandas.DataFrame with x, y, is_standard, and id columns
global ALL_SPEC_IDS
global ALL_SPECTRA
global MZ

tmp = GLOBAL_DATA.class_table
CLASS_DICT = {elem : list(tmp[elem]) for elem in tmp.columns} 
AVAILABLE_CLASSES = list(CLASS_DICT.keys())
print(AVAILABLE_CLASSES)

# Extract pairwise similarity matrices
SM_MS2DEEPSCORE = GLOBAL_DATA.ms2deepscore_sim
SM_MODIFIED_COSINE = GLOBAL_DATA.cosine_sim 
SM_SPEC2VEC = GLOBAL_DATA.spec2vec_sim
# Extract and expand t-sne df
TSNE_DF = GLOBAL_DATA.tsne_df
TSNE_DF["is_standard"] = GLOBAL_DATA.is_standard
TSNE_DF["id"] = GLOBAL_DATA.specxplore_id
# INITIALIZE COLOR_DICT
selected_class_data=CLASS_DICT[AVAILABLE_CLASSES[0]]
# INITIALIZE GRAYSCALE COLOUR MAPPING
n_colors=len(set(selected_class_data))
colors=data_transfer.construct_grey_palette(n_colors, white_buffer=20)
init_color_dict=data_transfer.create_color_dict(colors, selected_class_data)
# INITIALIZE ALL SPEC IDS LIST? NDARRAY
ALL_SPEC_IDS = GLOBAL_DATA.specxplore_id
# INITIALIZE ALL SPECTRA LIST
ALL_SPECTRA = GLOBAL_DATA.spectra
# CONSTRUCT SOURCE, TARGET AND VALUE ND ARRAYS
SOURCE, TARGET, VALUE = specxplore_data_cython.construct_long_format_sim_arrays(SM_MS2DEEPSCORE)
# CONSTRUCT MZ DATA LIST FOR VISUALIZATION
MZ = GLOBAL_DATA.mz

########################################################################################################################
# All settings panel
settings_panel = dbc.Offcanvas([
    html.P("SpecXplore defaults and limits can be modified here."),
    html.B("Set Edge Threshold:"),
    html.P("A value between 0 and 1 (excluded)"),
    dcc.Input(
        id="edge_threshold_input_id", type="number", 
        debounce=True, placeholder="Threshold 0 < thr < 1, def. 0.9", 
        style={"width" : "100%"}),
    html.B("Set expand level:"),
    html.P("A value between 1 and 5. Controls connection branching in EgoNet."),
    dcc.Input( id="expand_level_input", type="number", 
        debounce=True, 
        placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
        style={"width" : "100%"}),
    html.B("Select Class Level:"),
    dcc.Dropdown(
        id='class-dropdown' , multi=False, 
        clearable=False, options=AVAILABLE_CLASSES, 
        value=AVAILABLE_CLASSES[5]),
    html.B("Filter by Class(es):"),
    dcc.Dropdown(
        id='class-filter-dropdown' , multi=True, clearable=False, options=[],
        value=[]),
    ],
    id="offcanvas-settings",
    title="Settings Panel",
    is_open=False,)
########################################################################################################################
# Main selection panel
selection_panel = dbc.Offcanvas([
    html.P((
        "All selected spectrums ids from the overview graph:"
        "\n Selection can be modified here.")),
    dcc.Dropdown(id='specid-selection-dropdown', multi=True, 
            style={'width': '90%', 'font-size': "75%"}, 
            options=ALL_SPEC_IDS)],
    id="offcanvas-selection",
    placement="bottom",
    title="Selection Panel",
    is_open=False)
########################################################################################################################
# Focus selection panel
selection_focus_panel = dbc.Offcanvas([
    html.P((
        "All spectrum ids selected for focused analysis in network views."
        "\n Selection can be modified here.")),
    dcc.Dropdown(id='specid-focus-dropdown', multi=True, 
            style={'width': '90%', 'font-size': "100%"}, 
            options=ALL_SPEC_IDS),
    html.B("Enter spec_id for spectrum plot:"),
    dcc.Dropdown(id='specid-spectrum_plot-dropdown', multi=False, 
        options=ALL_SPEC_IDS, style={'width' : '50%'}),
    html.B("Enter spec_id1 (top) and spec_id2 (bottom) for mirrorplot:"),
    dcc.Dropdown(id='specid-mirror-1-dropdown', multi=False, 
        options=ALL_SPEC_IDS,
        style={'width' : '50%'}),
    dcc.Dropdown(id='specid-mirror-2-dropdown', multi=False, 
        options=ALL_SPEC_IDS,
        style={'width' : '50%'})],
    id="offcanvas-focus",
    placement="end",
    title="Focus Selection Panel",
    is_open=False)
########################################################################################################################
#button_style_logo = {'width': '25px', 'height': '50px', "fontSize": "30px", "textAlign": "center", "lineHeight": "25px",}
#button_style_text = {'width': '100px', 'height': '50px', "fontSize": "12px", "textAlign": "center", "lineHeight": "25px",}

button_style_logo = {'width': '40%', 'height': '11vh', "fontSize": "30px", "textAlign": "center", }
button_style_text = {'width': '60%', 'height': '11vh', "fontSize": "12px", "textAlign": "center", }
control_button_group = [
    dbc.ButtonGroup(
        [
        dbc.Button('⚙', id="btn-open-settings", style = 
                   {'width': '100%', 'height': '14vh', "fontSize": "25px"}),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks1-settings', style = button_style_logo),
            dbc.Button('EgoNet ', id='btn-run-egonet', n_clicks=0, style = button_style_text),
            ]),  
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks2-settings', style = button_style_logo),
            dbc.Button('ClustNet', id="btn-run-clustnet", n_clicks=0, style = button_style_text),  
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks3-settings', style = button_style_logo),  
            dbc.Button('Augmap', id="btn-run-augmap", n_clicks=0, style = button_style_text)
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks4-settings', style = button_style_logo),  
            dbc.Button('Spectrum Plot(s)', id='btn_push_spectrum', n_clicks=0, style = button_style_text)
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-fragmap-settings', style = button_style_logo),  
            dbc.Button('Fragmap', id="btn_push_fragmap", n_clicks=0, style = button_style_text)
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-meta-settings', style = button_style_logo),  
            dbc.Button('Metadata Table', id="btn_push_meta", n_clicks=0, style = button_style_text)
        ]),
        ],
        vertical=True,
    )]

    




########################################################################################################################
# good themes: VAPOR, UNITED ; see more:  https://bootswatch.com/
app=Dash(external_stylesheets=[dbc.themes.VAPOR])
app.layout=html.Div([
    # ROW 1: Title & hover response
    dbc.Row(
        [
        dbc.Col([html.H1([html.B("specXplore prototype")], style={"margin-bottom": "0em"})], width=6),
        dbc.Col([html.P("Hover Info: placeholder for cyto hover information", id = "hover-info-panel")])
        ]
    ),
    html.Br(),
    dbc.Row(
        [
        dbc.Col(control_button_group, width = 1),
        dbc.Col(
            [        
            # Remove overview plotly graph
            dcc.Graph(id="tsne-overview-graph", figure={}, style={"width":"100%","height":"80vh", "border":"1px grey solid"}),
            # empty cyto for id presence in layout before generation.
            html.Div([cyto.Cytoscape(id='cytoscape-tsne-subnet')], id='right-panel-tabs-content'),
            ], 
            width=11),
        ], 
        style={"margin": "0px"}, className="g-0"),
    html.Br(),
    dbc.Button("Open Selection", id="btn-open-selection", n_clicks=0),
    dbc.Button("Open Focus", id="btn-open-focus", n_clicks=0),
    settings_panel,
    selection_panel,
    selection_focus_panel,
    dcc.Store(id="edge_threshold", data=0.9),
    dcc.Store(id="expand_level", data=int(1)),
    dcc.Store(id="selected-filter-classes", data = []),
    dcc.Store(id="selected_class_level", data=AVAILABLE_CLASSES[0]),
    dcc.Store(id="selected_class_data", data=CLASS_DICT[AVAILABLE_CLASSES[0]]),
    dcc.Store(id="color_dict",  data=init_color_dict),
    html.Br(),
    dbc.Row([dbc.Col([html.Div(id="fragmap_panel", style={"width":"100%", "border":"1px grey solid"})], width=12)]),
    html.Br(),
    dbc.Row([dbc.Col([html.Div(id="metadata_panel", style={"width":"100%", "border":"1px grey solid"})], width=12)]),
    html.Br(),
    dbc.Row(
        [
        dbc.Col([html.Div(id="spectrum_plot_panel", style={"width":"100%", "border":"1px grey solid"})], width=6),
        dbc.Col([html.Div(id="mirrorplot_panel", style={"width":"100%", "border":"1px grey solid"})], width=6)
        ]
        ),
    ], 
    style={"width" : "100%"},
    )

# spectrum_plot_panel

@app.callback(
    Output("edge_threshold", "data"),
    Output("edge_threshold_input_id", "placeholder"),
    Input('edge_threshold_input_id', 'n_submit'),
    Input("edge_threshold_input_id", "value"))

def update_threshold_trigger_handler(n_submit, new_threshold):
    new_threshold, new_placeholder=data_transfer.update_threshold(new_threshold)
    return new_threshold, new_placeholder

@app.callback(
    Output("expand_level", "data"),
    Output("expand_level_input", "placeholder"),
    Input('expand_level_input', 'n_submit'),
    Input("expand_level_input", "value"))

def expand_trigger_handler(n_submit, new_expand_level):
    new_expand_level, new_placeholder=data_transfer.update_expand_level(
        new_expand_level)
    return new_expand_level, new_placeholder

@app.callback( ##################################################################################
    Output('selected-filter-classes', 'data'),
    Input('class-filter-dropdown', 'value')
)
def update_selected_filter_classes(values):
    return values

# GLOBAL OVERVIEW UPDATE TRIGGER
@app.callback(
    Output("tsne-overview-graph", "figure"), 
    Input("tsne-overview-graph", "clickData"), # or "hoverData"
    Input("selected_class_level", "data"),
    Input("selected_class_data", "data"),
    Input('selected-filter-classes', 'data'), 
    Input('specid-focus-dropdown', 'value'),
    State("color_dict", "data"),
    State("tsne-overview-graph", "figure"))

def left_panel_trigger_handler(
    point_selection, 
    selected_class_level, 
    selected_class_data, 
    class_filter_set,
    focus_selection,
    color_dict,
    figure_old):
    """ Modifies global overview plot in left panel """
    #global tsne_fig
    #print(type(figure_old))
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if (triggered_id == 'tsne-overview-graph' and point_selection):
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'selected_class_level':
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'selected_class_data':
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'selected-filter-classes':
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'specid-focus-dropdown' and focus_selection:
        print(f'Trigger element: {triggered_id}')
        tsne_fig=tsne_plotting.plot_tsne_overview(
            point_selection, selected_class_level, selected_class_data, TSNE_DF, 
            class_filter_set, color_dict, focus_selection)
        print(tsne_fig.data[0].marker.color)
    else:
        print("Something else triggered the callback.")
    tsne_fig=tsne_plotting.plot_tsne_overview(
        point_selection, selected_class_level, selected_class_data, TSNE_DF, 
        class_filter_set, color_dict, focus_selection)
    return tsne_fig

# CLASS SELECTION UPDATE ------------------------------------------------------
@app.callback(
    Output("selected_class_level", "data"), 
    Output("selected_class_data", "data"),
    Output("color_dict", "data"), 
    Output('class-filter-dropdown', 'options'), 
    Output('class-filter-dropdown', 'value'),
    Input("class-dropdown", "value"))
def class_update_trigger_handler(selected_class):
    """ Wrapper Function that construct class dcc.store data. """
    selected_class_data, color_dict=data_transfer.update_class(selected_class, 
        CLASS_DICT)
    print("Checkpoint - new selected class data constructed.")
    return selected_class, selected_class_data, color_dict, list(set(selected_class_data)), []

# RIGHT PANEL BUTTON CLICK UPDATES --------------------------------------------
@app.callback(
    Output('right-panel-tabs-content', 'children'),
    Input('btn-run-egonet', 'n_clicks'),
    Input('btn-run-augmap', 'n_clicks'),
    Input('btn-run-clustnet', 'n_clicks'),
    State('specid-selection-dropdown', 'value'), 
    State("color_dict", "data"),
    State("selected_class_data", "data"),
    State("edge_threshold", "data"),
    State("expand_level", "data"))
def right_panel_trigger_handler(
    n_clicks1, n_clicks2, n_clicks3, spec_id_selection, color_dict, 
    selected_class_data, threshold, expand_level):
    btn = ctx.triggered_id
    if btn == "btn-run-clustnet" and spec_id_selection:
        panel = clustnet.generate_cluster_node_link_diagram_cythonized(
            TSNE_DF, spec_id_selection, GLOBAL_DATA.ms2deepscore_sim, selected_class_data,
            color_dict, threshold, SOURCE, TARGET, VALUE, MZ)
    elif btn == "btn-run-egonet"  and spec_id_selection:
        panel = egonet.generate_egonet_cythonized(
            spec_id_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, 
            threshold, expand_level)
    elif btn == "btn-run-augmap"  and spec_id_selection:
        panel=augmap.generate_augmap_panel(
            spec_id_selection, GLOBAL_DATA.ms2deepscore_sim, GLOBAL_DATA.cosine_sim , GLOBAL_DATA.spec2vec_sim, 
            threshold)
    else:
        print("Nothing selected for display in right panel yet.")
        panel=[html.H6("empty-right-panel")]
    return panel



# tsne-overview selection data trigger ----------------------------------------
@app.callback(
    Output('specid-selection-dropdown', 'value'),
    Input('tsne-overview-graph', 'selectedData')
)
def plotly_selected_data_trigger(plotly_selection_data):
    """ Wrapper Function for tsne point selection handling. """
    if ctx.triggered_id == "tsne-overview-graph":
        selected_ids=data_transfer.extract_identifiers_from_plotly_selection(plotly_selection_data)
    else:
        selected_ids = []
    return selected_ids

# Fragmap trigger -------------------------------------------------------------
@app.callback(
    Output('fragmap_panel', 'children'),
    Input('btn_push_fragmap', 'n_clicks'), # trigger only
    State('specid-focus-dropdown', 'value'))
def fragmap_trigger(n_clicks, selection_data):
    """ Wrapper function that calls fragmap generation modules. """
    # Uses: global variable ALL_SPECTRA
    fragmap_panel=fragmap.generate_fragmap_panel(selection_data, ALL_SPECTRA)
    return fragmap_panel


@app.callback(
    Output("offcanvas-settings", "is_open"),
    Input("btn-open-settings", "n_clicks"),
    [State("offcanvas-settings", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output("offcanvas-selection", "is_open"),
    Input("btn-open-selection", "n_clicks"),
    [State("offcanvas-selection", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output("offcanvas-focus", "is_open"),
    Input("btn-open-focus", "n_clicks"),
    [State("offcanvas-focus", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('specid-focus-dropdown', 'value'),
    [Input('cytoscape-tsne-subnet', 'selectedNodeData')])
def displaySelectedNodeData(data):
    if data:
        focus_ids = []
        for elem in data:
            print(elem)
            focus_ids.append(int(elem["id"]))
        return focus_ids




# Metadata trigger ------------------------------------------------------------
@app.callback(
    Output('metadata_panel', 'children'),
    Input('btn_push_meta', 'n_clicks'), # trigger only
    State('specid-focus-dropdown', 'value'))
def metadata_trigger(n_clicks, selection_data):
    """ Wrapper function that calls fragmap generation modules. """
    # Uses: global variable ALL_SPECTRA
    if selection_data:
        tmpdf = TSNE_DF.iloc[selection_data]
        panel = dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in tmpdf.columns],
            data=tmpdf.to_dict("records"),
            style_cell=dict(textAlign="left"),
            style_header=dict(backgroundColor="magenta"),
            style_data=dict(backgroundColor="white"),
            sort_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},)
        return panel
    else:
        panel = [html.H6((
            "Select focus data and press " 
            "'Show Metdata' button for data table."))]
        return panel



@app.callback(
    Output('spectrum_plot_panel', 'children'),
    Input('specid-spectrum_plot-dropdown', 'value')) # trigger only
def generate_spectrum_plot(id):
    if id:
        fig = go.Figure(data=[go.Bar(
            x=[1, 2, 3, 5.5, 10],
            y=[10, 8, 6, 4, 2],
            width=[0.8, 0.8, 0.8, 3.5, 4] 
        )])
        fig.update_layout(title="Placeholder Graph spectrum_plot")
        panel = dcc.Graph(id = "spectrum_plot", figure= fig)
        return panel
    else: 
        return html.P("Select spectrum id for plotting.")


@app.callback(
    Output('mirrorplot_panel', 'children'),
    Input('specid-mirror-1-dropdown', 'value'),
    Input('specid-mirror-2-dropdown', 'value')) # trigger only
def generate_mirror_plot(spectrum_id_1, spectrum_id_2):
    if spectrum_id_1 and spectrum_id_2:
        fig = go.Figure(data=[
            go.Bar(
                x=[1, 2, 3, 5.5, 10],
                y=[10, 8, 6, 4, 2],
                width=[0.8, 0.8, 0.8, 3.5, 4] 
            )])
        fig.update_layout(title="Placeholder Graph mirrorplot")
        panel = dcc.Graph(id = "mirror_plot", figure= fig)
        return panel
    else: 
        return html.P("Select spectrum ids for plotting.")

if __name__ == '__main__':
    app.run_server(debug=True, port="8999")


                                                                               
