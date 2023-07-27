# Dark2 categorical color scale from color brewer; colorblind friendly
COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"]



# Netview parameters
NODE_SIZE = "10"
EDGE_SIZE = "1"

NETVIEW_STYLE = [
    {
        "selector": '.edge_within_set', 
        "style":{
            "line-color":"magenta",
              "background-color" : "magenta", 
              'opacity':0.6
        }
    }, 
    {
        "selector":'.edge_out_of_set', 
        "style":{
            "line-color":"black", 
            'opacity':0.4, 
            "background-color":"black",
            'border-width':EDGE_SIZE, 
            'border-color':'black'
        }
    },
    {
        "selector":'.node_out_of_set', 
        "style":{
            "line-color":"black", 
            'opacity':0.2, 
            "background-color":"black",
            'border-width':EDGE_SIZE, 
            'border-color':'black'
        }
    }
]

GENERAL_STYLE = [
    {
        'selector':'node', 
        'style': {
            'text-halign':'center', 
            'text-valign':'center', 
            'background-color': '#E5E4E2',
            'height':NODE_SIZE, 
            'width':NODE_SIZE, 
            "border-width":EDGE_SIZE, 
            "opacity":0.7
        }
    }, 
    {
        'selector':'.none',
        'style':{'color' : 'green'},
        'selector':'.is_highlighted', 
        'style': {
            'shape' : 'diamond', 
            'opacity':1, 
            'background-color': '#757573'
        }
    }
]

EDGE_STYLE = [
    {    
        'selector': 'edge',
        'style': {
            'width': 1 
        }
    }
]

SELECTED_NODES_STYLE = [
    {
        'selector': ':selected',
        'style': {
            'background-color': 'magenta',
            "border-color":"purple", 
            "border-width": 1,
            "border-style": "dashed"
        }
    }
]

SELECTED_STYLE_EGONET = [
    {
        'selector': ':selected',
        'style': {
            'background-color': 'magenta', 
            'label': 'data(label)', 
            "border-color":"purple",
            "border-style": "dashed"
        }
    }
]


BASIC_NODE_STYLE_SHEET_EGONET = [
    {
        'selector':'node', 
        'style': {
            'content':'data(label)',
            'text-halign':'center', 
            'text-valign':'center', 
            "shape":"circle",
            'height':NODE_SIZE, 
            'width':NODE_SIZE, 
            "border-width":EDGE_SIZE, 
            'opacity':0.2}
    }, 
    {
        'selector':'label', 
        'style':{
            'content':'data(label)',
            'color':'black', 
            "font-family": "Ubuntu Mono", 
            "font-size": "1px",
            "text-wrap": "wrap", 
            "text-max-width": 100
        }
    }
]

BASIC_EDGE_STYLE_EGONET = [
    {    
        'selector': 'edge',
        'style': {
            'width': 1 
        }
    }
]
