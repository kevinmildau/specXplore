# Utility functions for prototype dashboards.
import numpy as np
import re

# matchms data processing #####################################################
def extract_similarity_scores(sm):
    """Function extracts similarity matrix from matchms cosine scores array.
      The cosine scores similarity output of matchms stores output in a
      numpy array of pair-tuples, where each tuple contains 
      (sim, n_frag_overlap). This function extracts the sim scores, and returns
      a numpy array of identical size.
    """
    sim_data = [ ]
    for row in sm:
        for elem in row:
            sim_data.append(float(elem[0]))
    return(np.array(sim_data).reshape(sm.shape[0],sm.shape[1]))

def extract_n_fragment_overlap(sm):
    """Function extracts fragment overlap matrix from matchms cosine scores array.
      The cosine scores similarity output of matchms stores output in a
      numpy array of pair-tuples, where each tuple contains 
      (sim, n_frag_overlap). This function extracts the n_frag_overlap, and returns
      a numpy array of identical size.
    """
    sim_data = [ ]
    for row in sm:
        for elem in row:
            sim_data.append(int(elem[1]))
    return(np.array(sim_data).reshape(sm.shape[0],sm.shape[1]))


# Custom binning functions for spectrum raw data ###############################
def bin_spectrum(mz, intensity, bins):
    """ Function bins spectrum into specified bin ranges.
    
    Binning sums up the intensities of the joined fragments. Create bins using
    numpy.arange or numpy.linspace
    
    Parameters
    ----------------------------------------------------------------------------
    param1 : mz
        An array of mz values to be put into standardized bins.
    param2 : intensity
        An array of intensity values corresponding to the mz values. 
        Must be equal in length to mz array.
    param3 : bins
        An array of breakpoints defining the bins [begin, step1, step2, ..., end]
    
    Returns
    ----------------------------------------------------------------------------
    array : 
        An array of binned mz values with numeric data representing summed 
        intensities for the bin.
    """
    assert len(mz) == len(intensity), "mz and intensity arrays must be equal length!" 
    binned = np.histogram(mz, bins, weights = intensity)[0]
    return binned

if False:
    # bin_spectrum use case example:
    mz_values = np.array([1,4,56,58,70,90]) # A toy mz value observation
    intensities = np.array([1,1,1,2,3,1]) # toy intensities
    bins = np.arange(0,101, 10) # a range between 0 and 100 in steps of 10
    bin_spectrum(mz_values, intensities, bins)

def compute_adjacency_matrix(similarity_matrix, threshold):
    """ Computes Adjacency Array from similarity array using threshold.
    
    Assumes a np.array input with rows and columns containing values between 0 
    and 1, or -1 and 1. Evaluates whether any element in the array >= to the
    threshold, and if so replaces the value by 1, if not replaces the value by 
    0.
    
    Parameters
    ----------------------------------------------------------------------------
    param1 : similarity_matrix
        A square array of similarity values between 0 and 1 (or -1 and 1)
    param2 : threshold
        A threshold of similarity to be used as cutoff for edge creation. Repl-
        acement of score by 1 if above or equal, by 0 if below.

    Returns
    ----------------------------------------------------------------------------
    array : 
        An array containing 0 and 1 only. (dtype = int32)
    """
    return(np.where(similarity_matrix >= threshold, 1, 0).astype("int32"))

if False:
    compute_adjacency_matrix(np.array([[0.5,0.6], [0.6,0.5]]), 0.55)


