import plotly.graph_objects as go
import numpy as np
from typing import List
from specxplore.importing import Spectrum
from dash import dcc

PLOT_LAYOUT_SETTINGS = {
    'template':"simple_white",
    'xaxis':dict(title="Mass to charge ratio"),
    'yaxis':dict(title="Intensity", fixedrange=True),
    'hovermode':"x",
    'showlegend':False
}

def generate_single_spectrum_plot(spectrum : Spectrum) -> go.Figure:
    """ Generates single spectrum plot. """

    hover_trace_invisible = generate_bar_hover_trace(
        spectrum.mass_to_charge_ratios, 
        spectrum.intensities, 
        spectrum.spectrum_iloc
    )
    
    visual_trace = generate_bar_line_trace(
        spectrum.mass_to_charge_ratios, 
        spectrum.intensities
    )
    
    figure = go.Figure(
        data = [hover_trace_invisible] + visual_trace
    )
    figure.update_yaxes(range=[0, 1])
    figure.update_layout(
        title = {'text': f"Spectrum {spectrum.spectrum_iloc}"}, 
        **PLOT_LAYOUT_SETTINGS
    )
    return figure


def generate_bar_line_trace(x_values: np.ndarray, y_values: np.ndarray) -> List[go.Scatter]:
    """ Generates bar lines for arrays containing x and y values. """

    kwargs = {
        'mode': 'lines', 
        'opacity': 1, 
        'hoverinfo': "skip", 
        'name' : '', 
        'showlegend':False
    }
    lines_scatters = [
        go.Scatter(x = [x,x], y =  [0,y], meta=np.abs(y), marker={'color': "black"}, **kwargs) 
        for x, y 
        in zip(x_values, y_values)
    ]
    return lines_scatters


def generate_bar_hover_trace(x_values: np.ndarray, y_values: np.ndarray, spectrum_identifier : int) -> go.Scatter:
    """ Generates the hover trace information for each x, y value pair. """
    kwargs = {
        'mode': 'markers', 
        'opacity': 0, 
        'hovertemplate': "%{meta:.4f}<extra></extra>", 
        'name' : f'Spectrum ID = {spectrum_identifier}'
    }
    output_figure = go.Scatter(
        x=x_values, 
        y=y_values, 
        meta= np.abs(y_values),
        marker={'color': "black"}, 
        **kwargs
    )
    return output_figure


def generate_mirror_plot(top_spectrum : Spectrum, bottom_spectrum: Spectrum) -> go.Figure:
    """ Generates spectrum mirrorplot. """

    top_hover_trace_invisible = generate_bar_hover_trace(
        top_spectrum.mass_to_charge_ratios, 
        top_spectrum.intensities, 
        top_spectrum.spectrum_iloc
    )
    bottom_hover_trace_invisible = generate_bar_hover_trace(
        bottom_spectrum.mass_to_charge_ratios, 
        bottom_spectrum.intensities * -1.0, 
        bottom_spectrum.spectrum_iloc
    )
    top_visible_trace = generate_bar_line_trace(
        top_spectrum.mass_to_charge_ratios, 
        top_spectrum.intensities
    )
    bottom_visible_trace = generate_bar_line_trace(
        bottom_spectrum.mass_to_charge_ratios, 
        bottom_spectrum.intensities * -1.0
    )
    data = [top_hover_trace_invisible, bottom_hover_trace_invisible] + top_visible_trace + bottom_visible_trace

    figure = go.Figure(data = data)
    figure.update_yaxes(range=[-1, 1])
    figure.add_hline(y=0.0, line_width=1, line_color="black", opacity=1)
    figure.update_layout(
        title = {
            'text': f"Spectrum {top_spectrum.spectrum_iloc} vs Spectrum {bottom_spectrum.spectrum_iloc}"
        },
        **PLOT_LAYOUT_SETTINGS
    )
    return figure


def generate_multiple_spectra_figure_div_list(spectra : List[Spectrum]) -> List[dcc.Graph]:
    """ 
    Generate list of single spectrum plots, each of which embedded into a separate dcc.Graph containers.
    
    Parameters:
    -----------
    spectra : List[Spectrum]
        List of specxplore.data.Spectrum objects
    
    Returns:
    --------
        List of dcc.Graph objects; one spectrum plot per spectrum provided.
    """
    figure_div_list = []
    for spectrum in spectra:
        spectrum_figure = generate_single_spectrum_plot(spectrum)
        figure_div = dcc.Graph(
            id = f'_spectrum_plot_for_spectrum_{spectrum.spectrum_iloc}', 
            figure = spectrum_figure
        )
        figure_div_list.append(figure_div)
    return figure_div_list