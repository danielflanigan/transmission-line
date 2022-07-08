# transmission-line
Design and draw transmission lines in GDSII format using gdspy.

## What is it useful for?
This package was developed to draw GDSII design files to fabricate coplanar waveguide (CPW) superconducting resonators using optical lithography.
If the phase velocity of the transmission line is known, either from measurement or simulation, this allows the resonance frequencies to be set precisely by setting the correct length. 
It simplifies the calculation of the total length of a drawing of transmission sections with different geometries and that use different metals for the center trace and ground planes.

It would also be useful for drawing large arrays of resonators, such as kinetic inductance detector (KID) pixel arrays, for which modifying the geometry by hand is tedious and error-prone.

Follow these steps to draw a transmission line resonator with a particular total length:
1. Define a dummy outline that is close to the desired geometry. For example, the list of points `[(0, 0), (0, -200), (100, -200), (100, 0)]` form a "U" with sharp corners.
2. Create a subclass of `transmission_line.SmoothedSegment` (such as `cpw.CPW`) with that outline and the desired cross-sectional geometry. The corners that define the center will be replaced by circular arcs (avoiding problems with sharp edges), forming a "U" with rounded corners.
3. Use the `length` property of the instance to obtain the length of the transmission line including the bends, then create a object using a similar set of outline points with one leg extended or shortened by the difference between the actual and desired lengths.
4. Use the `draw` method of the instance either once (e.g., only the two gaps of a CPW) or several times (e.g. the center trace and ground planes on separate layers) to create the desired structures. 

## Requirements
The requirements are `setuptools-scm` (for installation) and `gdspy`, `numpy`, `scipy`, and `matplotlib` (used for rendering text). Building the docs requires `sphinx` and `sphinx-rtd-theme`. 

Some code may run in Python 2.7, but it is no longer supported.

## Contents
- `transmission_line.py` contains the core code.
- `trace.py` contains code to draw single wires, which could be used for microstrip or slotline.
- `cpw.py` contains code to draw co-planar waveguide transmission lines and to calculate properties such as inductance and capacitance per unit length.
- `mesh.py` contains code to calculate a mesh of points around transmission lines, useful for adding holes close to superconducting resonators.
