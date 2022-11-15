def extract_identifiers(plotly_selection_data):
    """ Function extracts custom_data id's from a provided point selection 
        dictionary. """
    if plotly_selection_data != None:
        selected_ids = [
            elem["customdata"][0] 
            for elem 
            in plotly_selection_data["points"]]
        print("slected _ids:", selected_ids)
    else:
        selected_ids = []
    return selected_ids