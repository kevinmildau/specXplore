# specXplore - Interactive and adjustable mass-spectral data exploration

# Contents

* [Overview](https://github.com/kevinmildau/specXplore#overview)
* [Installation guide - macos and linux](https://github.com/kevinmildau/specXplore#installation-guide---macos-and-linux)
* [Installation guide - WINDOWS](https://github.com/kevinmildau/specXplore#installation-guide---windows-in-development)
* [Workflow](https://github.com/kevinmildau/specXplore#workflow)

# Overview

SpecXplore is a python dashboard tool for adjustable LC-MS/MS spectral data exploration. It joins a t-SNE embedding that serves as an overview representation of mass spectral similarities based on ms2deepscore with interactive add-on and overlay representations providing further detailed information about the data. SpecXplore includes network views, similarity heatmaps, and fragmentation overview maps.

The specXplore workflow is separated into two stages. 
First, the user needs to process their spectral data in order to create a specXplore.SessionData object. This is done in interactive jupyter notebooks using the specXplore importing pipeline. The pipeline produces a specXplore session data object that is saved to the hard drive and can be fed directly into a specxplore dashboard session instance for visual exploration.

Before the specXplore workflow can be used, the package and its dependencies need to be installed using the guidelines below. Please not that the current version of specXplore works on Macos and Linux, but fails in windows.

## Installation Guide - Macos and Linux

To install and work with specXplore, we highly recommend using conda package management and will assume conda to be available in the guide below. 
To install specXplore, open the command line terminal and create a new conda environment with python version 3.8 as in the first code line below. 
Here, the conda environment is named specxplore_environment, but you are free to choose any name without spaces. 
Activate this environment using the second line of code.
Once successfully activated, install specXplore inside the corresponding environment using the final line of code. This will install all depenendencies required for running specXplore and may take a few minutes depending on existing package availability.

```{Bash}
conda create --name specxplore_environment python=3.8
conda activate specxplore_environment
pip install "git+https://github.com/kevinmildau/specXplore.git"
```

Note that depending on the operating system, developer tools including pip, python, and conda may need to be installed first. This will be the case if the console indicates that the conda or pip commands are not known. Please refer to [conda getting started](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) guide for information on how to set up conda. The pip installation may also fail because of a lack of C++ compilers which are not covered by the python package manager. For operating system specific instructions on how to set up compilers required by Cython, please refer to the [Cython Installation Guide](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).

## Installation Guide - WINDOWS

**WARNING:** The current version of specXplore **does not run** on Windows machines. Differences in how operating systems handle integer types cause importing pipeline function crashes that affect only Windows systems.

Installation of specxplore in windows will require the installation of ANACONDA for python environment management and its terminal, and Microsoft Visual C++ Redistributable for cython backend compulation. Unfortunately, the latter package involves a rather large installation taking up more than 7GB of space. The specXplore developers are not aware of any smaller installers for this in Windows. Once these two packages are installed, installation should work identically to the one in mac and linux systems from the **ANCONDA power shell**.

## Workflow

### Jupyter Notebook Pre-processing

One of the installed dependencies will be jupyter-notebook, which can be used to open the demo.ipynb book files to run any pre-processing or start the specXplore interactive dashboard. To open the demo notebook, make sure to use ```conda activate specxplore_environment``` to activate the environment with all specXplore dependencies installed, and run the ```jupyter-notebook``` command. Jupyter notebook will now open within the conda environment within which specxplore and all its dependencies are available. From here, navigate the jupyter graphical user interface towards a download of the [demo.ipynb jupyter notebook](https://github.com/kevinmildau/specXplore/blob/e601141c817a9ea8f9f0654957a718c7da80b8af/notebooks/demo.ipynb) file to run the example as provided or replace the demo data mgf filepath with your own data. Following the steps in this jupyter notebook allows the user to process their input data and run an interactive specxplore session. Note that specXplore currently requires a .mgf formatted file with MS/MS spectral data. 

To generate a .MGF file from your raw data please refer to processing options in your vendor specific software or the workflows described in MZmine [MZmine Getting Started](https://mzmine.github.io/mzmine_documentation/getting_started.html). MZmine3 provides [exporting options](https://mzmine.github.io/mzmine_documentation/module_docs/io/data-exchange-with-other-software.html#gnps-fbmniimn-export) for the .MGF file format. Feature lists should always contain some form of feature identifier, and specXplore expects the feature identifier key to be "feature_id". Renaming the feature identifying key in a .MGF file is possible using [matchms](https://matchms.readthedocs.io/en/latest/), specifically the [matchms.Spectrum module](https://matchms.readthedocs.io/en/latest/api/matchms.html#matchms.Spectrum) which provides a means of adding metadata keys to existing spectra in Python. A code example can be found [here](https://github.com/kevinmildau/specXplore/blob/e601141c817a9ea8f9f0654957a718c7da80b8af/notebooks/demo.ipynb). Alternatively, as a quick fix, any text editor may be used to replace any instance of the existing feature identifying key of format "wrong_key=" with "feature_id=". 

### Dashboard Use

To start the dashboard, follow the install guidelines above and then proceed to use the following code lines in the terminal:

```{bash}
conda activate specxplore_environment
python3
```

This will change the terminal to the python console. Within the python console, run the following two commands:

```{Python}
import specxplore.run_dashboard
specxplore.run_dashboard.app.run_server()
```

This will prompt a command line output specifying the following: "Dash is running on http://127.0.0.1:8050/". Open this link in your browser (Firefox tested) to open the empty specXplore dashboard. 
To load in data, open the settings panel and navigate to the final text input widget. 
Here, copy paste the full filepath (e.g. "/Users/janedoe/Documents/specxplore_session.pickle", each file explorer will have different options for obtaining this file path easily for a file) of the .pickle file containing the specxplore session data. 
Make sure that only the filepath is pasted, avoiding any quotation marks (i.e. '' or ""). 
The data will now be loaded into specXplore and can be interacted with. 
If the dataset looks highly compressed in the t-SNE overview figure with many nodes overlapping, make use of the scale input above the filepath input and increase the number to get updated scale informaiton. 
To quit specxplore, navigate to the console with the running server instance and press ctrl+c on the console (macos & linux). 
In addition, enter quit() in the then active python console to terminate the python process. 
Just closing the console also works.

# Dashboard Commands
Once the specXplore dashboard is there are a number of possible ways to interact with the visualizations. 

Clicking on a node in the t-SNE overview selects it. Starting a new selection may also reset previously triggered overlays. 
Hovering over a node will display node information in a textbox below the main t-SNE panel.
Using ctrl+mouse drag/click one can select more than one node in the t-SNE overview.
With appropriate node selections made, the various buttons can be used to trigger different overlay or add-on views. Overlay views are directly visualized on top of the t-SNE graph and disappear upon the next overlay visualization request. Add-on views are visualized below the t-SNE overview and the hover text box, and disappear upon the next add-on visualization request.

Changing settings in the settings panel does not immediately cause reruns of the open visualizations. Instead, the button has to be explicitly pressed again to redraw the respective visualization. Note that node selections are not altered when setting new settings, omitting any need to reselect nodes for redrawing.