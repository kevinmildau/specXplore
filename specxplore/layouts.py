from specxplore import identifiers
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_cytoscape

LAYOUT_STORE_CONTAINERS = html.Div(
    children= [
        dcc.Store(id=identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, data = None),
        dcc.Store(id=identifiers.STORE_EDGE_THRESHOLD, data=0.9),
        dcc.Store(id=identifiers.STORE_HOP_DISTANCE, data=int(1)),
        dcc.Store(id=identifiers.STORE_MAXIMUM_NODE_DEGREE, data = int(9999)),
        dcc.Store(id=identifiers.STORE_SELECTED_CLASS_LEVEL, data = None),
        dcc.Store(id=identifiers.STORE_CLASSES_TO_BE_HIGHLIGHTED, data = None),
        dcc.Store(id=identifiers.STORE_NODE_ELEMENTS, data = None),
        dcc.Store(id=identifiers.STORE_COLORBLIND_BOOLEAN, data = False),
    ]
)

SETTINGS_PANEL = dbc.Offcanvas(
    children=[
        html.B("Set Edge Threshold:"),
        html.P("A value between 0 and 1 (excluded)"),
        dcc.Input(
            id = identifiers.INPUT_EDGE_THRESHOLD, 
            type="number", 
            debounce=True, 
            placeholder="Threshold 0 < thr < 1, def. 0.9", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.Br(),
        html.Div(
            children=[
                dcc.Graph(
                    id = identifiers.FIGURE_EDGE_WEIGHT_HISTOGRAM, 
                    figure = [],  
                    style={
                        "width":"100%",
                        "height":"30vh"
                    }
                )
            ]
        ),
        html.Br(),
        html.B("Set maximum node degree:"),
        html.P("A value between 1 and 9999. The lower, the fewer edges are allowed for each node."),
        dcc.Input(
            id=identifiers.INPUT_MAXIMUM_NODE_DEGREE, 
            type="number", 
            debounce=True, 
            placeholder="1 <= Value <= 9999, def. 9999", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Higlight Class(es) by color:"),
        dcc.Dropdown(
            id=identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED , 
            multi=True, 
            clearable=False, 
            options=[],
            value=[]
        ),
        html.Br(),
        html.B("Set Hop Distance for EgoNet:"),
        html.P("A value between 1 and 5. Controls connection branching in EgoNet."),
        dcc.Input(
            id=identifiers.INPUT_HOP_DISTANCE, 
            type="number", 
            debounce=True, 
            placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Select Class Level:"),
        dcc.Dropdown( 
            id=identifiers.DROPDOWN_SELECT_CLASS_LEVEL, 
            multi=False, 
            clearable=False, 
            options = [], 
            value = []
        ),
        html.Br(),
        html.B("Toggle colorblind friendly for Augmap:"),
        daq.BooleanSwitch(
            id=identifiers.SWITCH_AUGMAP_COLORBLIND_CHANGE, 
            on=False
        ),
        html.Br(),
        html.B("Set top-K for fragmap"),
        html.P(
            children=(
                "A value between 1 and 9999. This value is used to restrict the number of"
                " fragments displayed in fragmap. "
                "If the number of fragments exceeds top-K, only the top-K highest intensity"
                " fragments are displayed " 
                "alongside the corresponding neutral losses."
            )
        ),
        dcc.Input(
            id=identifiers.INPUT_TOP_K_FRAGMENT_LIMIT_FOR_FRAGMAP, type="number", 
            debounce=True, value=25, placeholder = "25",
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Set Visal Scale:"),
        html.P(
            children=(
                "A value between 0 excluded and 2000. This value is used to expand the t-SNE layout to be"
                " larger or smaller."
            )
        ),
        dcc.Input(
            id=identifiers.INPUT_COORDINATE_SCALING, 
            type="number", 
            debounce=True, 
            value=400, 
            placeholder = "400",
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Session Data Location"),
        html.P("Provide path to a valid specXplore.pickle object."),
        html.Div(
            children=[
                dcc.Input(
                    id=identifiers.INPUT_UPLOAD_NEW_SESSION_DATA, 
                    type="text", 
                    placeholder=None, 
                    debounce=True)
            ]
        ),
        # Multi-line break to make sure that dcc dropdown interactions don't cause resizing issues
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    ],
    id=identifiers.PANEL_OFFCANVAS_SETTINGS,
    title="specXplore settings",
    is_open=False)

SELECTION_FOCUS_PANEL = dbc.Offcanvas(
    children=[
        html.P(
            children = (
                "All spectrum ids selected for focused analysis in network views."
                "\n Selection can be modified here."
            )
        ),
        dcc.Dropdown(
            id=identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, 
            multi=True, 
            style={"width": "90%", "font-size": "100%"}, 
            options=[])
    ],
    id=identifiers.PANEL_OFFCANVAS_FOCUS_SPECTRUM_ILOC_SELECTION,
    placement="end",
    title="Selection Panel",
    is_open=False)

LAYOUT_ROW_DETAILS_PANEL = dbc.Row(
    children = [
        dbc.Col(
            children = [
                html.Div(
                    id=identifiers.PANEL_DETAIL_VISUALIZATIONS, 
                    style={
                        "width":"100%", 
                        "height":"90vh", 
                        "border":"1px black solid"
                    }
                )
            ], 
            width=12)
    ], 
    className="g-0"
)

LAYOUT_ROW_MESSAGES_AND_HOVER = dbc.Row(
    children = [
        dbc.Col(
            children = [
                html.Tbody("Hover Info: placeholder for cyto hover information")
            ], 
            id = identifiers.PANEL_HOVER_INFO,
            style={"font-size" : "8pt", "border":"1px black solid"}),
        dbc.Col(
            children = [
                dcc.Markdown(
                    id=identifiers.PANEL_WARNING_MESSAGES_1, 
                    style={
                        "font-size" : "8pt", 
                        "border":"1px black solid"
                    }
                ),
                dcc.Markdown(
                    id=identifiers.PANEL_WARNING_MESSAGES_2, 
                    style={
                        "font-size" : "8pt", 
                        "border":"1px black solid"
                    }
                )
            ], 
            width=4), 
        dbc.Col(
            children = [
                html.Div(
                    id=identifiers.PANEL_NODE_DEGREE_LEGEND, 
                    style={
                        "width":"100%", 
                        "border":"1px black solid"
                    }
                )
            ], 
            width=8)
        ], 
    className="g-0"
)

BUTTON_HEIGHT = "9vh"
BUTTON_STYLE = {
    "width": "98%", 
    "height": BUTTON_HEIGHT, 
    "fontSize": "11px", 
    "textAlign": "center", 
    "border":"1px black solid",
    "backgroundColor" : "#8B008B"
}

LAYOUT_CONTROL_BUTTON_GROUP = [
    dbc.ButtonGroup(
        children=[
            dbc.Button("⚙ Settings", id=identifiers.BUTTON_OPEN_SETTINGS, style = BUTTON_STYLE),
            dbc.Button("View Selection", id=identifiers.BUTTON_OPEN_FOCUS, style = BUTTON_STYLE),
            dbc.Button("Show Node Degree ", id=identifiers.BUTTON_RUN_DEGREE_MAP, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("EgoNet ", id=identifiers.BUTTON_RUN_EGONET, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Show Edges for Selection", id=identifiers.BUTTON_RUN_NETVIEW, n_clicks=0, style = BUTTON_STYLE),  
            dbc.Button("Augmap", id=identifiers.BUTTON_RUN_AUGMAP, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Spectrum Plot(s)", id=identifiers.BUTTON_RUN_SPECPLOT, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Fragmap", id=identifiers.BUTTON_RUN_FRAGMAP, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Metadata Table", id=identifiers.BUTTON_RUN_METADATA_TABLE, n_clicks=0, style = BUTTON_STYLE)
        ],
        vertical=True, 
        style = {"width" : "100%"}
    )
]


LAYOUT_ROW_MAIN_PANEL_AND_BUTTONS = dbc.Row(
    children = [
        dbc.Col(LAYOUT_CONTROL_BUTTON_GROUP, width = 2),
        dbc.Col(
            children = [
                dash_cytoscape.Cytoscape( 
                    id=identifiers.CYTOSCAPE_MAIN_PANEL, 
                    elements = [], 
                    stylesheet = [],
                    layout={
                        "name":"preset", 
                        "animate":False, 
                        "fit": False}, 
                    boxSelectionEnabled=True,
                    style={
                        "width":"100%", 
                        "height":"80vh", 
                        "border":"1px black solid", 
                        "bg":"#feeff4", 
                        "minZoom":0.1, 
                        "maxZoom":2})
            ], 
            width=10
        ) 
    ], 
    style={"margin": "0px"}, 
    className="g-0"
)


LAYOUT_ROW_TITLE = dbc.Row(
    children = [
        dbc.Col(
            children = [
                html.H1(
                    children = [
                        html.B("specXplore")
                    ], 
                    style={"margin-bottom": "0em", "font-size" : "20pt"}
                )
            ], 
            width=6),
    ]
)

def initialize_layout(app : Dash) -> None:
    """ Initializes dash app layout for specxplore session. """

    app.layout = html.Div(
        children=[
            LAYOUT_ROW_TITLE,
            LAYOUT_ROW_MAIN_PANEL_AND_BUTTONS, 
            SETTINGS_PANEL,
            SELECTION_FOCUS_PANEL,
            LAYOUT_STORE_CONTAINERS,
            LAYOUT_ROW_MESSAGES_AND_HOVER,
            LAYOUT_ROW_DETAILS_PANEL,
        ], 
        style={"width" : "100%"},
    )
    return None