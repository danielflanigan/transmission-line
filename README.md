# transmission-line
Design and draw transmission lines in GDSII format using gdspy.

## Requirements
The requirements are `setuptools-scm` (for installation) and `gdspy`, `numpy`, `scipy`, and `matplotlib` (used for rendering text). Building the docs requires `sphinx` and `sphinx-rtd-theme`. 

Some of the code may run in Python 2.7, but it is no longer supported.

## Contents
- `transmission_line.py` contains the core code.
- `trace.py` contains code to draw single wires, which could be used for microstrip or slotline.
- `cpw.py` contains code to draw co-planar waveguide transmission lines and to calculate properties such as inductance and capacitance per unit length.
- `mesh.py` contains code to calculate a mesh of points around transmission lines, useful for adding holes close to superconducting resonators.
