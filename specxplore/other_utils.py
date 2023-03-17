import numpy as np

def standardize_array(array : np.ndarray):
    # mean centering and deviation scaling
    out = (array - np.mean(array)) / np.std(array)
    return out

def scale_array_to_minus1_plus1(array : np.ndarray) -> np.ndarray:
    """ Rescales array to lie between -1 and 1."""
    # Normalised [-1,1]
    out = 2.*(array - np.min(array))/np.ptp(array)-1
    return out