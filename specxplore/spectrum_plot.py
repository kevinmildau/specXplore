import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List
import specxplore.specxplore_data
from specxplore.specxplore_data import Spectrum
from dash import html, dcc

PLOT_LAYOUT_SETTINGS = {
        'template':"simple_white",
        'xaxis':dict(title="Mass to charge ratio"),
        'yaxis':dict(title="Intensity", fixedrange=True),
        'hovermode':"x",
        'showlegend':False}

def generate_single_spectrum_plot(spectrum : Spectrum) -> go.Figure:
    """ Generates single spectrum plot. """
    hover_trace_invisible = bar_hover_trace(spectrum.mass_to_charge_ratios, spectrum.intensities, spectrum.spectrum_iloc)
    visual_trace = bar_line_trace(spectrum.mass_to_charge_ratios, spectrum.intensities)
    figure = go.Figure(data = [hover_trace_invisible] + visual_trace)
    figure.update_yaxes(range=[0, 1])
    #visual_trace = bar_shape_trace(spectrum.mass_to_charge_ratios, spectrum.intensities)
    #figure.update_layout(shapes=visual_trace)
    figure.update_layout(title = {'text': f"Spectrum {spectrum.spectrum_iloc}"}, **PLOT_LAYOUT_SETTINGS)
    return figure

def bar_shape_trace(x_values: np.ndarray, y_values: np.ndarray) -> [go.layout.Shape]:
    kwargs = {'type': 'line', 'xref': 'x', 'yref': 'y', 'line_color': "black", 'line_width': 1, 'layer': "below",
              'opacity': 1}
    shapes = [go.layout.Shape(x0=x, y0=0, x1=x, y1=y, **kwargs) for x, y in zip(x_values, y_values)]
    return shapes

def bar_line_trace(x_values: np.ndarray, y_values: np.ndarray) -> List[go.Scatter]:
    kwargs = {
        'mode': 'lines', 'opacity': 1, 'hoverinfo': "skip", 
        'name' : '', 'showlegend':False}
    lines_scatters = [ go.Scatter(x = [x,x], y =  [0,y], meta=np.abs(y), marker={'color': "black"}, **kwargs) 
    for x,y in zip(x_values, y_values)]
    return lines_scatters

def bar_hover_trace(x_values: np.ndarray, y_values: np.ndarray, spectrum_identifier : int) -> go.Scatter:
    kwargs = {
        'mode': 'markers', 'opacity': 0, 'hovertemplate': "%{meta:.4f}<extra></extra>", 
        'name' : f'Spectrum ID = {spectrum_identifier}'}
    return go.Scatter(x=x_values, y=y_values, meta= np.abs(y_values), marker={'color': "black"}, **kwargs)

# Mirror plot of two spectra
def generate_mirror_plot(top_spectrum : Spectrum, bottom_spectrum: Spectrum) -> go.Figure:
    """ Generates spectrum mirrorplot. """
    top_hover_trace_invisible = bar_hover_trace(
        top_spectrum.mass_to_charge_ratios, top_spectrum.intensities, top_spectrum.spectrum_iloc)
    bottom_hover_trace_invisible = bar_hover_trace(
        bottom_spectrum.mass_to_charge_ratios, bottom_spectrum.intensities * -1.0, bottom_spectrum.spectrum_iloc)
    top_visible_trace = bar_line_trace(top_spectrum.mass_to_charge_ratios, top_spectrum.intensities)
    bottom_visible_trace = bar_line_trace(
        bottom_spectrum.mass_to_charge_ratios, bottom_spectrum.intensities * -1.0)
    data = [top_hover_trace_invisible, bottom_hover_trace_invisible] + top_visible_trace + bottom_visible_trace
    figure = go.Figure(data = data)
    figure.update_yaxes(range=[-1, 1])
    figure.add_hline(y=0.0, line_width=1, line_color="black", opacity=1)
    figure.update_layout(
        title = {'text': f"Spectrum {top_spectrum.spectrum_iloc} vs Spectrum {bottom_spectrum.spectrum_iloc}"},
        **PLOT_LAYOUT_SETTINGS)
    return figure

# DEVELOPER NOTE:
# Initially attempted a single plotly object. However, make_subplot from plotly created many layout related issues, and
# would have required tedious workarounds to accomodate our multiple traces per subplot scenario. dcc.Graph listing
# ultimately appeared simpler.
def generate_multiple_spectra_figure_div_list(spectra : List[Spectrum]) -> List[dcc.Graph]:
    """ Generate list of single spectrum plots, each of which embedded into a separate dcc.Graph containers.
    """
    figure_div_list = []
    for spectrum in spectra:
        spectrum_figure = generate_single_spectrum_plot(spectrum)
        figure_div = dcc.Graph(id = f'_spectrum_plot_for_spectrum_{spectrum.spectrum_iloc}', figure = spectrum_figure)
        figure_div_list.append(figure_div)
    return figure_div_list

def generate_mirrorplot_panel(spec_id_1, spec_id_2, all_spectra):
    if False:
    # if spec_id_1 and spec_id_2:
        fig = mirror_plot(id_a=spec_id_1, id_b=spec_id_2, spec_list=all_spectra)
        out = [
            html.Div(
                dcc.Graph(
                    id = "mirrorplot-panel", figure=fig, 
                    style={"width":"50%","height":"60vh", 
                        "border":"1px grey solid"}))]
        return(out)
    else:
        out = [
            html.H6(("Two spectra need to be selected for mirrorplot."))
        ]
        return(out)

def generate_specplot_panel(spec_id, all_spectra):
    if False:
    #if spec_id:
        fig = spectrum_plot(id=spec_id, spec_list=all_spectra)
        out = [
            html.Div(
                dcc.Graph(
                    id = "specplot-panel", figure=fig, 
                    style={"width":"50%","height":"60vh", 
                        "border":"1px grey solid"}))]
        return(out)
    else:
        out = [
            html.H6(("One spectrum needs to be selected for mirrorplot."))
        ]
        return(out)