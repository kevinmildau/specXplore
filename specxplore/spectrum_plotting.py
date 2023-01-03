# Developer Notes:

# This module is not yet done. Pending import of plotting functions from Henry.

from dash import html, dcc

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