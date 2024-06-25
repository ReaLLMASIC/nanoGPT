# PyPPA - PPA Analyzer Flow In Python
PyPPA is a Python library for running Power Performance and Area (PPA) analysis and is based on [OpenFASoC](https://github.com/idea-fasoc/OpenFASOC) and the [OpenROAD flow scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts). Planned features include switchable EDA tools, sweeping ranges of hyperparameters, optimization, and parallelized flows.

### Getting Started
PyPPA is entirely written in Python, except for the TCL scripts called by the EDA tools, and the platform (PDK).

#### Installation
1. Install [Python 3](https://www.python.org/).
2. Install [Yosys](https://github.com/YosysHQ/yosys) and [OpenROAD](https://github.com/the-OpenROAD-Project/openroad) by either of the following methods:
	- OpenFASoC [installation guide](https://openfasoc.readthedocs.io/en/latest/getting-started.html) but it includes extra tools.
	- Since PyPPA is based on the OpenROAD flow scrips (ORFS), its [documentation](https://openroad-flow-scripts.readthedocs.io/en/latest/user/UserGuide.html) can be followed but this also includes Klayout.
	- Build from source: [Yosys documentation](https://yosyshq.net/yosys/download.html), [OpenROAD documentation](https://openroad.readthedocs.io/en/latest/user/Build.html).