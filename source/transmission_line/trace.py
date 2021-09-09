""""This module contains classes for drawing lone traces with no other structures present.

These classes could be used in the positive sense, where structures represent metal, as microstrip (with a separate
ground plane), or in the negative sense, where structures represent the absence of metal, as slotline.
"""
from __future__ import absolute_import, division, print_function

import gdspy
import numpy as np

from transmission_line.transmission_line import POINTS_PER_DEGREE, MAX_POINTS, Segment, SmoothedSegment, to_point


class Trace(SmoothedSegment):
    """A single wire with bends rounded to avoid sharp corners."""

    def __init__(self, outline, trace, radius=None, points_per_degree=POINTS_PER_DEGREE, round_to=None):
        """Instantiate without drawing.

        :param iterable[indexable] outline: the outline points, before smoothing; see :class:`SmoothedSegment`.
        :param float trace: the width of the trace.
        :param radius: if None, use twice the width; see :func:`smooth`.
        :type radius: float or None
        :param float points_per_degree: the default is POINTS_PER_DEGREE; see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        if radius is None:
            radius = 2 * trace
        super(Trace, self).__init__(outline=outline, radius=radius, points_per_degree=points_per_degree,
                                    round_to=round_to)

    def draw(self, cell, origin, layer=0, datatype=0, start_overlap=0, end_overlap=0, max_points=MAX_POINTS,
             gdsii_path=False):
        """Draw this trace into the given cell as a GDSII polygon (or path) and return the drawn object.

        By default, :class:`gdspy.FlexPath` draws a polygon; to draw a path (including a zero-width path),
        use `gdsii_path=True`. The overlap regions, whose length is not counted by :meth:`Segment.length`,
        are passed to :class:`gdspy.FlexPath` as `ends=(start_overlap, end_overlap)`.

        :param cell: the cell into which to draw the trace, if not None.
        :type cell: gdspy.Cell or None
        :param point origin: the point at which to place the start of the trace.
        :param int layer: the GDSII layer.
        :param int datatype: the GDSII datatype.
        :param float start_overlap: the overlap length at the start.
        :param float end_overlap: the overlap length at the end.
        :param int max_points: polygons with more than this number of points are fractured.
        :param bool gdsii_path: passed to :class:`gdspy.FlexPath`; if True, create and return a GDSII path; if False,
                                convert the path to a Polygon.
        :return: the drawn object.
        :rtype: tuple[gdspy.PolygonSet or gdspy.FlexPath]
        """
        points = [to_point(origin) + point for point in self.points]
        if gdsii_path:
            element = gdspy.FlexPath(points=points, width=self.trace, layer=layer, datatype=datatype,
                                     max_points=max_points, ends=(start_overlap, end_overlap), gdsii_path=True)
        else:
            element = gdspy.FlexPath(points=points, width=self.trace, layer=layer, datatype=datatype,
                                     max_points=max_points, ends=(start_overlap, end_overlap)).to_polygonset()

        if cell is not None:
            cell.add(element=element)
        return (element,)


class TraceTransition(Segment):
    """Transition between two Traces with different widths.

    The points of this class are (start_point, end_point). It draws a single polygon in the shape of a trapezoid.
    """

    def __init__(self, start_point, start_trace, end_point, end_trace, round_to=None):
        """Instantiate without drawing in any cell.

        The points of this structure are (start_point, end_point).

        :param indexable start_point: the start point of the transition.
        :param float start_trace: the trace width of the previous section.
        :param indexable end_point: the end point of the transition.
        :param float end_trace: the trace width of the following section.
        :param float round_to: see :class:`Segment`.
        """
        super(TraceTransition, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_trace = start_trace
        self.end_trace = end_trace

    def draw(self, cell, origin, layer=0, datatype=0):
        """Draw this structure into the given cell and return the one drawn polygon.

        :param cell: the cell into which to draw the transition, if not None.
        :type cell: gdspy.Cell or None
        :param point origin: the points of the structure are relative to this point.
        :param int layer: the GDSII layer.
        :param int datatype: the GDSII datatype.
        :return: the drawn polygon.
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
        if cell is not None:
            cell.add(element=polygon)
        return (polygon,)
