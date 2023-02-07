import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data, Spectrum
import numpy as np
from specxplore import spectrum_plot
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
with open("data_import_testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle).spectra
spectra = [Spectrum(spec.peaks.mz, max(spec.peaks.mz),idx, spec.peaks.intensities) for idx, spec in enumerate(data)]


figure01 = spectrum_plot.generate_single_spectrum_plot(spectra[0])
#figure.show(renderer = "browser")

figure02 = spectrum_plot.generate_mirror_plot(spectra[14], spectra[16])
#figure.show(renderer = "browser")

div_list = spectrum_plot.generate_multiple_spectra_figure_div_list(spectra[14:17])
#figure.show(renderer = "browser")


# Initialise the app
app = dash.Dash(__name__)
# Define the app
app.layout = html.Div([
        html.Div(html.H1("Single Spectrum")),
        html.Div(dcc.Graph(id = "single-plot", figure=figure01)),
        html.Div(html.H1("Mirror Plot")),
        html.Div(dcc.Graph(id = "mirror-plot", figure=figure02)),
        html.Div(html.H1("Multiple Single Spectra")),
        html.Div(div_list, id="Multiple-Single-Spectra")]
)
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)