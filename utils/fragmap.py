from dash import html

def generate_fragmap(spec_ids, all_spectra):
    if spec_ids and len(spec_ids) >= 2:
        # TODO: incorporate Henry's fragmap scripts.
        out = [html.H6("Fragmap in development.")]
    else:
        out = [html.H6("Select focus data and press generate fragmap button for fragmap.")]
    return