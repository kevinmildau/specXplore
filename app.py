# Main specXplore prototype
from logging import warning
import dash
from dash import dcc, html, ctx, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from specxplore import visuals as visual_utils
from specxplore import egonet
from specxplore import augmap
from specxplore import tsne_plotting
from specxplore import cytoscape_cluster
from specxplore import fragmap
from specxplore import parsing

from specxplore import cython_utils

import pickle
import dash_cytoscape as cyto
import plotly.graph_objects as go
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data

#app=Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

with open("testing/results/phophe_specxplore.pickle", 'rb') as handle:
    specxplore_data = pickle.load(handle) 


global STRUCTURE_DICT
global CLASS_DICT
#STRUCTURE_DICT, CLASS_DICT=load_utils.process_structure_class_table(
#    "data/classification_table.csv")

#CLASS_DICT=load_utils.extract_classes_from_ms2query_results(
#    "data-input/results/GNPS-NIH-NATURALPRODUCTSLIBRARY.csv")
#print(list(CLASS_DICT.keys())[0])
#print(CLASS_DICT[list(CLASS_DICT.keys())[4]])
tmp = specxplore_data.class_table
CLASS_DICT = {elem : list(tmp[elem]) for elem in tmp.columns}

#print(CLASS_DICT)
global AVAILABLE_CLASSES
#AVAILABLE_CLASSES=list(CLASS_DICT.keys())
#print(AVAILABLE_CLASSES)
AVAILABLE_CLASSES = list(CLASS_DICT.keys())
print(AVAILABLE_CLASSES)

global SM_MS2DEEPSCORE
global SM_MODIFIED_COSINE
global SM_SPEC2VEC
#SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC=load_utils.load_pairwise_sim_matrices()
SM_MS2DEEPSCORE = specxplore_data.ms2deepscore_sim
SM_MODIFIED_COSINE = specxplore_data.cosine_sim 
SM_SPEC2VEC = specxplore_data.spec2vec_sim

global TSNE_DF
#with open("data/tsne_df.pickle", 'rb') as handle:
#    TSNE_DF=pickle.load(handle)
#    print(TSNE_DF)
TSNE_DF = specxplore_data.tsne_df

TSNE_DF["is_standard"] = specxplore_data.is_standard
TSNE_DF["id"] = specxplore_data.specxplore_id
#print(TSNE_DF)
# tmp TSNE_DF modification to trial standard highlighting
#TSNE_DF["is_standard"] = False
#TSNE_DF.iloc[2:40, TSNE_DF.columns.get_loc('is_standard')] = True

# Initializing color dict
selected_class_data=CLASS_DICT[AVAILABLE_CLASSES[0]]
print(selected_class_data)
# Create overall figure with color_dict mapping
n_colors=len(set(selected_class_data)) # TODO: speed this up using n_clust argument that is pre-computed
colors=visual_utils.construct_grey_palette(n_colors, white_buffer=20)
init_color_dict=visual_utils.create_color_dict(colors, selected_class_data)

global ALL_SPEC_IDS
#ALL_SPEC_IDS=TSNE_DF.index # <-- add list(np.unique(spec_id_list of sorts))
ALL_SPEC_IDS = specxplore_data.specxplore_id
print(ALL_SPEC_IDS[0:3])

global ALL_SPECTRA
#file=open("data/cleaned_demo_data.pickle", 'rb')
#ALL_SPECTRA=pickle.load(file)
#file.close()
ALL_SPECTRA = specxplore_data.spectra

# PROTOTYPING CYTHON DATA STRUCTURES
SOURCE, TARGET, VALUE = cython_utils.construct_long_format_sim_arrays(SM_MS2DEEPSCORE)

print(SOURCE[0:3], VALUE[0:3])

#file = open("data/extracted_precursor_mz_values.pickle", 'rb')
#MZ = pickle.load(file) 
#file.close()
MZ = specxplore_data.mz
print(MZ[0:3])

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


app=dash.Dash(external_stylesheets=[dbc.themes.YETI]) # MORPH or YETI style.
app.layout=html.Div([
    dbc.Row([
        dbc.Col([html.H1([html.B("specXplore prototype")], 
            style={"margin-bottom": "-0.1em"})], width=4)]),
    html.Br(),
    dbc.Row([
        dbc.Col(
            [dcc.Graph(
                id="tsne-overview-graph", figure={}, 
                style={"width":"100%","height":"80vh", 
                "border":"1px grey solid"})], 
            width=6),
        dbc.Col([
            html.Div(
                [cyto.Cytoscape(id='cytoscape-tsne-subnet')], # <- empty cyto for id presence in layout before generation.
                id='right-panel-tabs-content')], # <------ currently the only one.
            #html.Div(id='right-panel-tabs-content-2', style= {'display': 'none'}),
            #html.Div(id='right-panel-tabs-content-3', style= {'display': 'none'})], 
            width=6),
    ], style={"margin-bottom": "-1em"}),
    html.Br(),
        dbc.Button("Open Settings", id="btn-open-settings", n_clicks=0),
        dbc.Button("Open Selection", id="btn-open-selection", n_clicks=0),
        dbc.Button("Run EgoNet", id="btn-run-egonet", n_clicks=0),
        dbc.Button("Run ClustNet", id="btn-run-clustnet", n_clicks=0),
        dbc.Button("Run AugMap", id="btn-run-augmap", n_clicks=0),
        dbc.Button("Open Focus", id="btn-open-focus", n_clicks=0),
        dbc.Button("Generate Fragmap", id="btn_push_fragmap"),
        dbc.Button("Show Metadata", id="btn_push_meta"),
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
    dbc.Row([dbc.Col([html.Div(id="fragmap_panel", 
        style={"width":"100%", "border":"1px grey solid"})], width=12)]),
    html.Br(),
    dbc.Row([dbc.Col([html.Div(id="metadata_panel", 
        style={"width":"100%", "border":"1px grey solid"})], width=12)]),
    html.Br(),
    dbc.Row([
        dbc.Col([html.Div(id="spectrum_plot_panel", 
            style={"width":"100%", "border":"1px grey solid"})], width=6),
        dbc.Col([html.Div(id="mirrorplot_panel", 
            style={"width":"100%", "border":"1px grey solid"})], width=6)]),
    ], 
    style={"width" : "100%"},
    )

# spectrum_plot_panel


@app.callback([Output("edge_threshold", "data"),
               Output("edge_threshold_input_id", "placeholder")],
              [Input('edge_threshold_input_id', 'n_submit'),
              Input("edge_threshold_input_id", "value")])

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
        #selected_point=point_selection["points"][0]["customdata"][0]
        #color_dict[selected_class_data[selected_point]]="#FF10F0"
        #selected_class = selected_class_data[selected_point]
        #for key in color_dict.keys():
        #    current_class = key
        #    if current_class == selected_class:
        #        tsne_fig.update_traces(
        #            marker=dict(color='#FF10F0'), 
        #            selector={'name': selected_class})
        #    else:
        #        tsne_fig.update_traces(
        #            marker=dict(color=color_dict[current_class]), 
        #            selector={'name': current_class})
        #return tsne_fig
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
        #tsne_fig.data[0].marker.color = ['red', 'blue', 'green']
        

    else:
        print("Something else triggered the callback.")
    
    #if figure_old.keys():
    #    print(figure_old["layout"].keys())
    #    print(len(figure_old["data"]))
    #    print(figure_old['data'][0].keys())
    
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
    selected_class_data, color_dict=parsing.update_class(selected_class, 
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
        panel = cytoscape_cluster.generate_cluster_node_link_diagram_cythonized(
            TSNE_DF, spec_id_selection, SM_MS2DEEPSCORE, selected_class_data,
            color_dict, threshold, SOURCE, TARGET, VALUE, MZ)
        #panel=cytoscape_cluster.generate_cluster_node_link_diagram(
        #    TSNE_DF, spec_id_selection, SM_MS2DEEPSCORE, selected_class_data, 
        #    color_dict, threshold)
    elif btn == "btn-run-egonet"  and spec_id_selection:
        panel = egonet.generate_egonet_cythonized(
            spec_id_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, 
            threshold, expand_level)
        #panel=egonet.generate_egonet(
        #    spec_id_selection, SM_MS2DEEPSCORE, TSNE_DF, threshold, expand_level)
    elif btn == "btn-run-augmap"  and spec_id_selection:
        panel=augmap.generate_augmap_panel(
            spec_id_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, 
            threshold)
    else:
        warning("Nothing selected for display in right panel yet.")
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
        selected_ids=parsing.extract_identifiers(plotly_selection_data)
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
    app.run_server(debug=True)


                                                                               
