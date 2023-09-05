# specXplore - Interactive and adjustable mass-spectral data exploration

# Contents

* [Overview](https://github.com/kevinmildau/specXplore-prototype#overview)
* [Installation guide - UNIX](https://github.com/kevinmildau/specXplore-prototype#installation-guide---unix)
* [Installation guide - WINDOWS](https://github.com/kevinmildau/specXplore-prototype#installation-guide---windows)
* [Workflow](https://github.com/kevinmildau/specXplore-prototype#workflow)
* [Developer Notes](https://github.com/kevinmildau/developer-notes)

# Overview
The specXplore workflow is separated into two stages. First, the user needs to process their spectral data in order to create a specXplore.SessionData object. This is done in interactive jupyter notebooks using spectral importing and processing using matchms, and session data creation using specXplores inbuilt methods. The session data object is then saved as a .pickle object to be loaded into the dashboard. Before the specXplore workflow can be used, the package and its dependencies need to be installed.

## Installation Guide - UNIX

To install specXplore, create a new conda environment with python version 3.8 as in the first code line below. Here, the conda environment is named specxplore_environment, but you are free to choose any name without spaces. Activate this environment using the second line of code, and use pip to install specxplore from github. Note that depending on the operating system, developer tools including pip, python, and conda may need to be installed first. This will be the case if the console indicates that the conda or pip commands are not known. The pip installation may also fail because of a lack of C++ compilers which are not covered by the python package manager. For operating system specific instructions on how to set up compilers required by Cython, please refer to the [Cython Installation Guide](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).

```{Bash}
conda create --name specxplore_environment python=3.8
conda activate specxplore_environment
pip install "git+https://github.com/kevinmildau/specXplore-prototype.git"

```
## Installation Guide - WINDOWS

```diff
**WARNING**
``` 
The current version of specXplore does not run on windows machines. Differences in how operating systems handle integer types cause the main view panels to be non-responsive. 

Installation of specxplore in windows requires the installation of ANACONDA for python environment management and its terminal, and Microsoft Visual C++ Redistributable for cython backend compulation. Unfortunately, the latter package involves a rather large installation taking up more than 7GB of space. The specXplore developers are not aware of any smaller installs for this in Windows. Once these two packages are installed, installation should work identically to the one in UNIX systems from the ANCONDA terminal.

To call up python in the ANCONDA terminal, run the following command omitting the '3' that would be added in macos unix systems:

```{bash}
conda activate specxplore_environment
python
```

Dashboard use in windows subsequently works as described in [Workflow](https://github.com/kevinmildau/specXplore-prototype#workflow).


## Workflow

### Jupyter Notebook Pre-processing

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

This will run specXplore as a local development server that provides all specXplore functionalities. To load in data, open the settings panel and navigate to the final text input widget. Here, copy paste the full filepath (e.g. "/Users/janedoe/Documents/specxplore_session.pickle", each file explorer will have different options for obtaining this file path easily for a file) of the .pickle file containing the specxplore session data. Make sure that only the filepath is pasted, avoiding any quotation marks (i.e. '' or "").  To quit specxplore, navigate to the console with the running server instance and press ctrl+c on the console (UNIX). In addition, enter quit() in the then active python console to terminate the python process. Just closing the console also works.


# Developer Notes

* When opening the dashboard, the initial dashboard will be empty. Any triggered callbacks will cause some form no-data error to be shown in the console, this can safely be ignored.
* To circumvent costly json conversions required by dash callbacks, specxplore makes use of a global variable to store the session data and update it. This global variable is part of the python process started with the terminal. To properly end a specXplore session, the python console needs to be closed as well. 