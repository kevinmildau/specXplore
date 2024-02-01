# specXplore - Interactive and adjustable mass-spectral data exploration

# Contents

* [Overview](https://github.com/kevinmildau/specXplore?tab=readme-ov-file#overview)
* [Installation guide - macos and linux](https://github.com/kevinmildau/specXplore?tab=readme-ov-file#installation-guide---macos-and-linux)
* [Installation guide - WINDOWS](https://github.com/kevinmildau/specXplore?tab=readme-ov-file#installation-guide---windows)
* [Workflow](https://github.com/kevinmildau/specXplore?tab=readme-ov-file#workflow)

# Overview

SpecXplore is a python dashboard tool for adjustable LC-MS/MS spectral data exploration. It joins a t-SNE embedding that serves as an overview representation of mass spectral similarities based on ms2deepscore with interactive add-on and overlay representations providing further detailed information about the data. SpecXplore includes network views, similarity heatmaps, and fragmentation overview maps.

The specXplore workflow is separated into two stages. 
First, the user needs to process their spectral data in order to create a specxplore session data object. This is done in interactive Jupyter notebooks using the specXplore importing pipeline. The pipeline produces a specXplore session data object that is saved to the hard drive and can be fed directly into a specxplore dashboard session instance for visual exploration.

Before the specXplore workflow can be used, the package and its dependencies need to be installed using the guidelines below. Please note that the current version of specXplore works on Macos and Linux but fails in windows.

## Installation Guide - Macos and Linux

**Warning:** users making use of macos arm64 computers should be aware of issue 199 for ms2deepscore https://github.com/matchms/ms2deepscore/issues/199. The current ms2deepscore package version may lead to ms2deepscore similarity predictions that are not in accordance with results on other systems (windows, ubuntu, macos intel). This issue does not result in any errors or warning messages, but makes ms2deepscore results unreliable! This directly affects the specXplore importing pipeline when making use of default ms2deepscore similarity scores on the affected systems. The model files and similarity predictions are working on other systems.

To install and work with specXplore, we highly recommend using conda package management and will assume conda to be available in the guide below. 
To install specXplore, open the command line terminal and create a new conda environment with python version 3.8 as in the first code line below. 
Here, the conda environment is named specxplore_environment, but you are free to choose any name without spaces. 
Activate this environment using the second line of code.
Once successfully activated, install specXplore inside the corresponding environment using the final line of code. This will install all dependencies required for running specXplore and may take a few minutes depending on existing package availability.

```{Bash}
conda create --name specxplore_environment python=3.8
conda activate specxplore_environment
pip install "git+https://github.com/kevinmildau/specXplore.git"
```

Note that depending on the operating system, developer tools including pip, python, and conda may need to be installed first. This will be the case if the console indicates that the conda or pip commands are not known. Please refer to [conda getting started](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) guide for information on how to set up conda. The pip installation may also fail because of a lack of C++ compilers which are not covered by the python package manager. For operating system specific instructions on how to set up compilers required by Cython, please refer to the [Cython Installation Guide](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).

To run ms2query, ms2deepscore, and spec2vec, model and library files are required. Pre-trained models are available via ms2query for both [positive](https://zenodo.org/records/10527997) and [negative](https://zenodo.org/records/10528030) mode data. Model and library files for positive or negative mode should be put into separate folders. The importing pipeline requires the appropriate model files to function.

## Installation Guide - WINDOWS

Installation of specxplore in windows will require the installation of ANACONDA for python environment management and its terminal, and [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022) for cython backend computation (see [Cython Installation Guide](https://cython.readthedocs.io/en/latest/src/quickstart/install.html)). Once these two packages are installed, installation should work identically to the one in mac and linux systems from the **ANCONDA power shell**.

## Workflow

### Jupyter Notebook Pre-processing

One of the installed dependencies will be jupyter-notebook, which can be used to open the demo.ipynb book files to run any pre-processing or start the specXplore interactive dashboard. To open the demo notebook, make sure to use ```conda activate specxplore_environment``` to activate the environment with all specXplore dependencies installed, and run the ```jupyter-notebook``` command. Jupyter notebook will now open within the conda environment within which specxplore and all its dependencies are available. From here, navigate the Jupyter graphical user interface towards a download of the [demo.ipynb Jupyter notebook](https://github.com/kevinmildau/specXplore/blob/master/notebooks/demo.ipynb) file to run the example as provided or replace the demo data mgf filepath with your own data. Following the steps in this Jupyter notebook allows the user to process their input data and run an interactive specxplore session. 

Note that specXplore currently requires a .mgf formatted file with MS/MS spectral data. To generate a .MGF file from your raw data please refer to processing options in your vendor specific software or the workflows described in MZmine [MZmine Getting Started](https://mzmine.github.io/mzmine_documentation/getting_started.html). MZmine3 provides [exporting options](https://mzmine.github.io/mzmine_documentation/module_docs/io/data-exchange-with-other-software.html#gnps-fbmniimn-export) for the .MGF file format. Feature lists should always contain some form of feature identifier, and specXplore expects the feature identifier key to be "feature_id". Renaming the feature identifying key in a .MGF file is possible using [matchms](https://matchms.readthedocs.io/en/latest/), specifically the [matchms.Spectrum module](https://matchms.readthedocs.io/en/latest/api/matchms.html#matchms.Spectrum) which provides a means of adding metadata keys to existing spectra in Python. A code example can be found [here](https://github.com/kevinmildau/specXplore/blob/master/notebooks/example-feature-id-processing.ipynb). Alternatively, as a quick fix, any text editor may be used to replace any instance of the existing feature identifying key of format "wrong_key=" with "feature_id=". 

# Dashboard Use Commands
Once the specXplore dashboard is opened, there are a number of possible ways to interact with the visualizations. 

Clicking on a node in the t-SNE overview selects it. Starting a new selection may also reset previously triggered overlays. 
Hovering over a node will display node information in a textbox below the main t-SNE panel.
Using ctrl+mouse drag/click one can select more than one node in the t-SNE overview.
With appropriate node selections made, the various buttons can be used to trigger different overlay or add-on views. Overlay views are directly visualized on top of the t-SNE graph and disappear upon the next overlay visualization request. Add-on views are visualized below the t-SNE overview and the hover text box, and disappear upon the next add-on visualization request.

Changing settings in the settings panel does not immediately cause reruns of the open visualizations. Instead, the button must be explicitly pressed again to redraw the respective visualization. Note that node selections are not altered when setting new settings, omitting any need to reselect nodes for redrawing.