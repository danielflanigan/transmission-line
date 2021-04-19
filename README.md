# transmission-line
Design and draw transmission lines in GDSII format using gdspy.

## Requirements
The requirements are `setuptools-scm` (for installation) and `gdspy`, `numpy`, and `scipy`. Building the docs requires `sphinx` and `sphinx-rtd-theme`. 

Python 2.7 is currently supported, and will be dropped eventually.

## Contents
- `transmission_line.py` contains the core code. 
- `cpw.py` contains code to draw co-planar waveguide transmission lines and to calculate properties such as inductance and capacitance per unit length.
- `mesh.py` contains code to calculate a mesh of points around transmission lines, useful for adding holes close to superconducting resonators.
