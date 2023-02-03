# Imports --------------------------------------------------------------------------------------------------------------
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matchms
import pickle


# Functions -----------------------------------------------------------------------------------------------------------


# Conversion of matchms objects to pandas
def spectrum_list_to_pandas(id_list: [int], spec_list: [matchms.Spectrum]) -> pd.DataFrame:

    # Ensure the provided ID list is valid
    assert all([ind in range(0, len(spec_list)) for ind in id_list]), \
        f"Error: Provided ID list {id_list} contains ID's which do not match the spec_list of length {len(spec_list)}."

    # Initialize empty list of data frames
    spec_data = list()

    # Iterate over all indices specified or all indices in the data
    for identifier in id_list:

        # Extra data, create pandas data frame, and append to growing list data frames
        spec_data.append(pd.DataFrame({"spectrum": identifier,
                                       "m/z": spec_list[identifier].mz,
                                       "intensity": spec_list[identifier].intensities}))

    # Return single long data frame
    return pd.concat(objs=spec_data, ignore_index=True)


# Bar-plot of spectrum
def spectrum_plot(id: int, spec_list: [matchms.Spectrum]) -> go.Figure:

    # Subset selected ID and convert to pandas
    spectrum = spectrum_list_to_pandas(id_list=[id], spec_list=spec_list)

    # Create Empty Figure
    figure = go.Figure(data=bar_hover_trace(spectrum["m/z"], spectrum["intensity"]))

    # Set X and Y Axes Scales
    figure.update_yaxes(range=[0, 1])

    # Add each bar of the bar-plot as a line shape
    figure.update_layout(shapes=bar_shape_trace(spectrum["m/z"], spectrum["intensity"]))

    # Format Layout
    figure.update_layout(
        template="simple_white",
        xaxis=dict(title="m/z"),
        yaxis=dict(title="Intensity", fixedrange=True),
        hovermode="x",
        showlegend=False
    )

    # Return
    return figure


def bar_shape_trace(x_values: [float], y_values: [float]) -> [go.layout.Shape]:
    kwargs = {'type': 'line', 'xref': 'x', 'yref': 'y', 'line_color': "black", 'line_width': 1, 'layer': "below",
              'opacity': 1}
    shapes = [go.layout.Shape(x0=x, y0=0, x1=x, y1=y, **kwargs) for x, y in zip(x_values, y_values)]
    return shapes


def bar_hover_trace(x_values: [float], y_values: [float]) -> go.Scatter:
    kwargs = {'mode': 'markers', 'opacity': 0, 'hovertemplate': "%{meta:.4f}<extra></extra>"}
    return go.Scatter(x=x_values, y=y_values, meta=y_values.abs(), marker={'color': "black"}, **kwargs)


# Mirror plot of two spectra
def mirror_plot(id_a: int, id_b: int, spec_list: [matchms.Spectrum]) -> go.Figure:

    # Subset selected ID's and convert to pandas
    spectrum_a = spectrum_list_to_pandas(id_list=[id_a], spec_list=spec_list)
    spectrum_b = spectrum_list_to_pandas(id_list=[id_b], spec_list=spec_list)

    # Create Mirror Plot
    figure = go.Figure(
        data=[
            bar_hover_trace(spectrum_a["m/z"], spectrum_a["intensity"]),
            bar_hover_trace(spectrum_b["m/z"], spectrum_b["intensity"] * (-1.0))
        ]
    )

    #
    # Add each bar of the bar-plot as a line shape
    figure.update_layout(shapes=bar_shape_trace(spectrum_a["m/z"], spectrum_a["intensity"]) +
                                bar_shape_trace(spectrum_b["m/z"], spectrum_b["intensity"] * (-1.0))
                         )

    # Ensure the axes are scaled properly
    figure.update_yaxes(range=[-1, 1])

    # Add reference y-axis (at y=0)
    figure.add_hline(
        y=0.0,
        line_width=1,
        line_color="black",
        opacity=1
    )

    # Format Layout
    figure.update_layout(
        template="simple_white",
        xaxis=dict(title="m/z"),
        yaxis=dict(title="Intensity", fixedrange=True),
        hovermode="x",
        showlegend=False
    )

    # Return
    return figure


# Main ----------------------------------------------------------------------------------------------------------------

# Load Original Spectra
file = open("./data/0009_spectrums_difficult_1.pickle", 'rb')
spectra = list(pickle.load(file))
file.close()

# Create and display spectrum plot
single_spectrum_figure = spectrum_plot(id=1, spec_list=spectra)
single_spectrum_figure.show()

# Create and display mirror spectrum plot
mirror_plot_figure = mirror_plot(id_a=3, id_b=4, spec_list=spectra)
mirror_plot_figure.show()
