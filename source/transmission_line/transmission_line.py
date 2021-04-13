"""This module contains the core classes and functions.

Internally, the package format for a point is a numpy ndarray with shape (2,). This simplifies mathematical
operations that treat points as vectors in two dimensions. Functions and methods that require an input value to be a
'point' will accept any indexable object with indicies 0 and 1. (Any elements with higher indices are ignored.)
Inputs are converted to package format using :func:`to_point`, and sequences of points are converted to package
format using :func:`to_point_list`.
"""
from __future__ import absolute_import, division, print_function

import gdspy
import numpy as np

# This is the maximum number of points in a polygon allowed by the GDSII specification.
GDSII_POLYGON_MAX_POINTS = 8190
DEFAULT_POINTS_PER_RADIAN = 60


def to_point(indexable):
    """Return a package-format point, a numpy ndarray with shape (2,) containing (x, y) coordinates.

    :param point indexable: an indexable object with integer indices 0 and 1, such as a two-element tuple.
    :return: an array with shape (2,) containing the values at these two indices.
    :rtype: numpy.ndarray
    """
    return np.array([indexable[0], indexable[1]])


def to_point_list(iterable):
    """Return a list of package-format points from using :func:`to_point` on each element of the given iterable.

    This function accepts, for example, a list of two-element tuples.

    :param iterable[point] iterable: an iterable of indexable objects that all have integer indices 0 and 1.
    :return: a list of arrays with shape ``(2,)`` containing (x, y) coordinates.
    :rtype: list[numpy.ndarray]
    """
    return [to_point(point) for point in iterable]


def from_increments(increments, origin=(0, 0)):
    """Return a list of points starting from the given origin and separated by the given increments.

    Specify a path in terms of the differences between points instead of the absolute values::

        >>> from_increments(increments=[(200, 0), (0, 300)], origin=(100, 0))
        [np.array([100, 0]), np.array([300, 0]), np.array([300, 300])]

    :param iterable[point] increments: the differences between consecutive returned points.
    :param point origin: the starting point of the list.
    :return: a list of points in the package format.
    :rtype: list[numpy.ndarray]
    """
    points = [to_point(origin)]
    for increment in [to_point(point) for point in increments]:
        points.append(points[-1] + increment)
    return points

# ToDo: update docstrings below here


# ToDo: warn if consecutive points are too close together to bend properly.
def smooth(points, radius, points_per_radian=DEFAULT_POINTS_PER_RADIAN):
    """Return a list of smoothed points constructed by adding points to change the given corners into arcs.

    At each corner, the original point is replaced by points that form circular arcs that are tangent to the original
    straight sections. The returned path will not contain any of the given points except for the starting and ending
    points, because all of the interior points will be replaced.

    If the given radius is too large there is no way to make this work, and the results will be ugly. **No warning is
    currently given when this happens, so you need to inspect your design.** The given radius should be smaller than
    about half the length of the shorted straight section. If several points lie on the same line, the redundant ones
    are removed.

    :param points: a list of points in package format.
    :type points: iterable[indexable]
    :param float radius: the radius of the circular arcs used to connect the straight segments.
    :param int points_per_radian: the number of points per radian of arc; the default of 60 (about 1 per degree) is
                                  usually enough.
    :return: a list of smoothed points.
    :rtype: list[numpy.ndarray]
    """
    bends = []
    angles = []
    corners = []
    offsets = []
    for before, current, after in zip(points[:-2], points[1:-1], points[2:]):
        before_to_current = current - before
        current_to_after = after - current
        # The angle at which the path bends at the current point, in (-pi, pi)
        bend_angle = np.angle(np.inner(before_to_current, current_to_after) +
                              1j * np.cross(before_to_current, current_to_after))
        if np.abs(bend_angle) > 0:  # If the three points are co-linear then drop the current point
            # The distance from the corner point to the arc center
            h = radius / np.cos(bend_angle / 2)
            # The absolute angle of the arc center point, in (-pi, pi)
            theta = (np.arctan2(before_to_current[1], before_to_current[0]) +
                     bend_angle / 2 + np.sign(bend_angle) * np.pi / 2)
            # The offset of the arc center relative to the corner
            offset = h * np.array([np.cos(theta), np.sin(theta)])
            # The absolute angles of the new points (at least two), using the absolute center as origin
            arc_angles = (theta + np.pi + np.linspace(-bend_angle / 2, bend_angle / 2,
                                                      int(np.ceil(np.abs(bend_angle) * points_per_radian) + 1)))
            bend = [current + offset + radius * np.array([np.cos(phi), np.sin(phi)]) for phi in arc_angles]
            bends.append(bend)
            angles.append(bend_angle)
            corners.append(current)
            offsets.append(offset)
    return bends, angles, corners, offsets


class SegmentList(list):
    """A list subclass for Segments that are joined sequentially to form a path."""

    # def draw(self, cell, origin, positive_layer, negative_layer, result_layer):
    def draw(self, cell, origin, *args, **kwargs):
        """Draw all of the Segments contained in this SegmentList into the given cell, connected head to tail.

        The Segments are drawn so that the origin of each segment after the first is the end of the previous segment.

        :param cell: The cell into which the result is drawn.
        :type cell: Cell
        :param origin: The point to use for the origin of the first Segment.
        :type origin: indexable
        :param args: arguments passed to :meth:`Segment.draw`.
        :param kwargs: keyword arguments passed to :meth:`Segment.draw`.
        :return: None.
        """
        # It's crucial to avoiding input modification that this also makes a copy.
        point = to_point(origin)
        for segment in self:
            segment.draw(cell, point, *args, **kwargs)
            # NB: using += produces an error when casting int to float.
            point = point + segment.end

    def __getitem__(self, item):
        """Re-implement this method so that slices return SegmentLists."""
        if isinstance(item, slice):
            return self.__class__(super().__getitem__(item))
        else:
            return super().__getitem(item)

    @property
    def start(self):
        """The start point of the SegmentList."""
        return self[0].start

    @property
    def end(self):
        """The end point of the SegmentList."""
        return np.sum(np.vstack([element.end for element in self]), axis=0)

    @property
    def span(self):
        """The difference between start and end points: span = end - start, in the vector sense."""
        return self.end - self.start

    @property
    def length(self):
        """The sum of the lengths of the Segments in this SegmentList."""
        return np.sum([element.length for element in self])


class Segment(object):
    """An element of a SegmentList that can draw itself into a cell."""

    def __init__(self, points, round_to=None):
        """The given points are saved as :attr:`_points`, and should generally not be modified.

        :param points: the points that form the Segment.
        :type points: iterable[indexable]
        :param round_to: if not None, the coordinates of each point are rounded to this value; useful for ensuring that
                         all the points in a design lie on a grid (larger than the database unit size).
        :type round_to: float, int, or None
        """
        points = to_point_list(points)
        if round_to is not None:
            points = [round_to * np.round(p / round_to) for p in points]
        self._points = points

    @property
    def points(self):
        """The points (``list[numpy.ndarray]``) in this Segment, rounded to ``round_to`` (read-only)."""
        return self._points

    @property
    def start(self):
        """The start point (``numpy.ndarray``) of the Segment (read-only)."""
        return self._points[0]

    @property
    def end(self):
        """The end point (``numpy.ndarray``) of the Segment (read-only)."""
        return self._points[-1]

    @property
    def x(self):
        """A ``numpy.ndarray`` containing the x-coordinates of all points (read-only)."""
        return np.array([point[0] for point in self.points])

    @property
    def y(self):
        """A ``numpy.ndarray`` containing the y-coordinates of all points (read-only)."""
        return np.array([point[1] for point in self.points])

    @property
    def length(self):
        """The length of the Segment, calculating by adding the lengths of straight lines connecting the points."""
        return np.sum(np.hypot(np.diff(self.x), np.diff(self.y)))

    def draw(self, cell, origin, **kwargs):
        """Draw this Segment in the given cell.

        Subclasses implement this method to draw themselves.

        :param Cell cell: the cell into which this segment will be drawn.
        :param indexable origin: draw the segment relative to this point; the meaning depends on the segment type.
        :return: None.
        """
        pass


class SmoothedSegment(Segment):
    """An element of a SegmentList that can draw itself into a cell, with corners smoothed using :func:`smooth`."""

    def __init__(self, outline, radius, points_per_radian, round_to=None):
        """The given outline points are passed to :func:`smooth` and the result is stored in the instance attributes
        :attr:`bends`, :attr:`angles`, :attr:`corners`, and :attr:`offsets`.

        :param iterable[indexable] outline: the outline points, before smoothing.
        :param float radius: the radius of the circular arcs used to connect the straight segments; see :func:`smooth`.
        :param int points_per_radian: the number of points per radian of arc; see :func:`smooth`.
        :param round_to: if not None, the coordinates of each outline point are rounded to this value **before
                         smoothing**; useful for ensuring that all the points in a design lie on a grid larger than the
                         database unit size.
        :type round_to: float or None
        """
        super(SmoothedSegment, self).__init__(points=outline, round_to=round_to)
        self.radius = radius
        self.points_per_radian = points_per_radian
        self.bends, self.angles, self.corners, self.offsets = smooth(self._points, radius, points_per_radian)

    @property
    def points(self):
        """The smoothed points (``list[numpy.ndarray]``); the outline points are ``_points`` (read-only)."""
        p = [self.start]
        for bend in self.bends:
            p.extend(bend)
        p.append(self.end)
        return p


class AbstractTransmissionLine(object):
    """An abstract transmission line.

    Subclasses should implement the properties given here to calculate inductance and capacitance per unit length.
    """

    @property
    def capacitance_per_unit_length(self):
        """Return the capacitance per unit length in farads per meter."""
        pass

    @property
    def inductance_per_unit_length(self):
        """Return the inductance per unit length in henries per meter."""
        pass

    @property
    def characteristic_impedance(self):
        """Return the characteristic impedance in ohms."""
        return (self.inductance_per_unit_length / self.capacitance_per_unit_length) ** (1 / 2)

    @property
    def phase_velocity(self):
        """Return the phase velocity in meters per second."""
        return (self.inductance_per_unit_length * self.capacitance_per_unit_length) ** (-1 / 2)


class Trace(SmoothedSegment):
    """A single wire.

    This class could be used in the positive sense, where structures represent metal, as microstrip (with a separate
    ground plane), or in the negative sense, where structures represent the absence of metal, as slotline.

    It can be drawn to overlap at either end with the adjacent elements, and the overlap lengths are not counted when
    calculating the total length. This is useful to avoid double-counting when an electrical connection is formed by an
    overlap.
    """

    def __init__(self, outline, trace, start_overlap=0, end_overlap=0, radius=None,
                 points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """
        :param iterable[indexable] outline: the outline points, before smoothing; see :class:`SmoothedSegment`.
        :param float trace: the width of the trace.
        :param float start_overlap: the overlap length at the start.
        :param float end_overlap: the overlap length at the end.
        :param radius: if None, use twice the width; see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: the default is DEFAULT_POINTS_PER_RADIAN; see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.start_overlap = start_overlap
        self.end_overlap = end_overlap
        if radius is None:
            radius = 2 * trace
        super(Trace, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
                                    round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0, max_points=GDSII_POLYGON_MAX_POINTS, **flexpath_keywords):
        """Draw this trace into the given cell as a GDSII polygon (or path) and return the drawn object.

        :class:`gdspy.FlexPath` by default draws a polygon; to draw a path, pass `gdsii_path=True`. The overlap
        regions, whose length is not counted by `self.length()`, are passed to :class:`gdspy.FlexPath` as
        `ends=(start_overlap, end_overlap)`.

        :param gdspy.Cell cell: the cell into which to draw the trace.
        :param point origin: the point at which to place the start of the trace.
        :param int layer: the layer on which to draw the trace.
        :param int datatype: the GDSII datatype.
        :param int max_points: polygons with more than this number of points are fractured by gdspy; the default value
                               overrides the gdspy default of 199.
        :param flexpath_keywords: keywords passed directly to :class:`gdspy.FlexPath`.
        :return: the drawn object.
        :rtype: gdspy.FlexPath
        """
        points = [to_point(origin) + point for point in self.points]
        flexpath = gdspy.FlexPath(points=points, width=self.trace, layer=layer, datatype=datatype,
                                  max_points=max_points, ends=(self.start_overlap, self.end_overlap),
                                  **flexpath_keywords)
        cell.add(element=flexpath)
        return flexpath


class TraceTransition(Segment):
    """Transition between two Traces with different widths.

    The points of this class are (start_point, end_point). It draws a single polygon in the shape of a trapezoid.
    """

    def __init__(self, start_point, start_trace, end_point, end_trace, round_to=None):
        """Instantiate without drawing in any cell.

        The points of this structure are (start_point, end_point).

        :param point start_point: the start point of the transition, typically the end point of the previous section.
        :param start_trace: the trace width of the previous section.
        :param end_point: the end point of the transition, typically the start point of the following section.
        :param end_trace: the trace width of the following section.
        :param round_to: see :class:`Segment`.
        """
        super(NegativeCPWTransitionBlank, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_trace = start_trace
        self.start_gap = start_gap
        self.end_trace = end_trace
        self.end_gap = end_gap

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the one drawn polygon.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the polygon that was drawn into the cell.
        :rtype: gdspy.Polygon
        """
        v = self.end - self.start
        phi = np.arctan2(v[1], v[0])
        rotation = np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
        points = [(0, self.start_trace / 2),
                  (self.length, self.end_trace / 2),
                  (self.length, -self.end_trace / 2),
                  (0, -self.start_trace / 2)]
        points_rotated = [np.dot(rotation, to_point(p).T).T for p in points]
        points_rotated_shifted = [to_point(origin) + self.start + p for p in points_rotated]
        polygon = gdspy.Polygon(points=points_rotated_shifted, layer=layer, datatype=datatype)
        cell.add(element=polygon)
        return polygon
