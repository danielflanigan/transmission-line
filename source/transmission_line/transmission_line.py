"""This module contains the core classes and functions.

Internally, the package format for a point is a :class:`numpy.ndarray` with shape (2,). This simplifies calculations
that treat points as vectors in two dimensions. To avoid forcing users to input points in this format, user-facing
functions and methods convert single input points to package format using :func:`to_point`, and convert sequences of
points using :func:`to_point_list`. The docstrings for such functions that accept two-element indexable objects
specify this using type 'indexable' and specify iterables of such objects using type 'iterable[indexable]'. If the
inputs must already be converted to package format, and to describe return types in package format, the documentation
uses 'point' and 'list[point]'.
"""
from __future__ import absolute_import, division, print_function
import warnings

import gdspy
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import numpy as np

# The maximum number of polygon vertices in the original GDSII specification is 200. This is the (conservative) default
# value used by gdspy.
SAFE_GDSII_POLYGON_POINTS = 199

# The maximum number of polygon vertices allowed by the GDSII file structure is 8191. Some GDSII readers may open files
# that contain more points, but, anecdotally, some will fail to load such designs. This value is the package default.
MAX_GDSII_POLYGON_POINTS = 8190

# Both of the limits above are one less than the maximum, possibly because the vertex that closes the polygon is stored
# in the file but does not have to be passed to gdspy, which closes polygons automatically.
# The value below is the default used when drawing polygons throughout this package.
MAX_POINTS = MAX_GDSII_POLYGON_POINTS
# It can be changed for a single `draw` call by passing a different value of `max_points` or it can be changed globally:
# from transmission_line import transmission_line as tl
# tl.MAX_POINTS = tl.SAFE_GDSII_POLYGON_POINTS  # For use with old GDSII readers
# tl.MAX_POINTS = 0  # gdspy treats 0 as infinity, and will not fracture polygons to reduce the number of vertices.

# This is the default number of points per degree of arc used by :func:`smooth` to bend transmission lines.
POINTS_PER_DEGREE = 1

# These are the default font properties used by :func:`polygon_text`, which renders text with matplotlib.
FONT_PROPERTIES = {
    'family': 'sans-serif',
    'style': 'normal'
}


def to_point(indexable):
    """Return a package-format point, a :class:`numpy.ndarray` with shape (2,) containing (x, y) coordinates.

    :param indexable indexable: an indexable object with integer indices 0 and 1, such as a two-element tuple.
    :return: an array with shape (2,) containing the values at these two indices.
    :rtype: point
    """
    return np.array([indexable[0], indexable[1]])


def to_point_list(iterable):
    """Return a list of package-format points from using :func:`to_point` on each element of the given iterable.

    This function accepts, for example, a list of two-element tuples, or a single ndarray with shape ``(N, 2)``, as used
    by gdspy for polygons.

    :param iterable[indexable] iterable: an iterable of indexable objects that all have integer indices 0 and 1.
    :return: a list of arrays with shape ``(2,)`` containing (x, y) coordinates.
    :rtype: list[point]
    """
    return [to_point(point) for point in iterable]


def from_increments(increments, origin=(0, 0)):
    """Return a list of points starting from the given origin and separated by the given increments, treated as vectors.

    Specify a path in terms of the differences between points instead of the absolute values::

        >>> from_increments(increments=[(200, 0), (0, 300)], origin=(100, 0))
        [np.array([100, 0]), np.array([300, 0]), np.array([300, 300])]

    :param iterable[indexable] increments: the vector differences between consecutive points.
    :param point origin: the starting point.
    :return: a list of points in package format.
    :rtype: list[point]
    """
    points = [to_point(origin)]
    for increment in to_point_list(increments):
        points.append(points[-1] + increment)
    return points


def polygon_text(text, size, position, layer=0, datatype=0, font_properties=None, tolerance=0.1):
    """Return the given text as polygons.

    If the text size is small, holes in letters can create structures that do not lift off easily. To avoid this,
    one option is to pass `family='stencil'`, which draws the text in an uppercase-only stencil font (available on
    Windows) with no holes.

    :param str text: the text to render as polygons.
    :param float size: the approximate size of the text in user units.
    :param indexable position: the coordinates are (left_edge, baseline), so the text may descend below the baseline.
    :param int layer: the GDSII layer.
    :param int datatype: the GDSII datatype.
    :param font_properties: if None, use :attr:`FONT_PROPERTIES` in this module; if dict, update these defaults;
                            see :module:`matplotlib.font_manager` for valid keys.
    :type font_properties: dict or None
    :param float tolerance: this has something to do with the number of points used to draw the polygon; the default
                            seems fine.
    :return: polygons representing the text.
    :rtype: gdspy.PolygonSet
    """
    fp = FONT_PROPERTIES.copy()
    if font_properties is not None:
        fp.update(font_properties)
    polygons = _render_text(text=text, size=size, position=position, font_prop=FontProperties(**fp),
                            tolerance=tolerance)
    return gdspy.PolygonSet(polygons=polygons, layer=layer, datatype=datatype)


def _render_text(text, size=None, position=(0, 0), font_prop=None, tolerance=0.1):
    """This function is copied from https://gdspy.readthedocs.io/en/stable/gettingstarted.html#using-system-fonts"""
    path = TextPath(position, text, size=size, prop=font_prop)
    polys = []
    xmax = position[0]
    for points, code in path.iter_segments():
        if code == path.MOVETO:
            c = gdspy.Curve(*points, tolerance=tolerance)
        elif code == path.LINETO:
            c.L(*points)
        elif code == path.CURVE3:
            c.Q(*points)
        elif code == path.CURVE4:
            c.C(*points)
        elif code == path.CLOSEPOLY:
            poly = c.get_points()
            if poly.size > 0:
                if poly[:, 0].min() < xmax:
                    i = len(polys) - 1
                    while i >= 0:
                        if gdspy.inside(
                                poly[:1], [polys[i]], precision=0.1 * tolerance
                        )[0]:
                            p = polys.pop(i)
                            poly = gdspy.boolean(
                                [p],
                                [poly],
                                "xor",
                                precision=0.1 * tolerance,
                                max_points=0,
                            ).polygons[0]
                            break
                        elif gdspy.inside(
                                polys[i][:1], [poly], precision=0.1 * tolerance
                        )[0]:
                            p = polys.pop(i)
                            poly = gdspy.boolean(
                                [p],
                                [poly],
                                "xor",
                                precision=0.1 * tolerance,
                                max_points=0,
                            ).polygons[0]
                        i -= 1
                xmax = max(xmax, poly[:, 0].max())
                polys.append(poly)
    return polys


# ToDo: warn if consecutive points are too close together to bend properly.
def smooth(points, radius, points_per_degree=POINTS_PER_DEGREE, already_package_format=False):
    """Return a list of smoothed points constructed by adding points to change the given corners into circular arcs.

    At each corner, the original point is replaced by points that form circular arcs that are tangent to the original
    straight sections. Because this process replaces all of the interior points, the returned path will not contain any
    of the given points except for the start and end.

    If the given radius is too large compared to the length of the straight sections there is no way to make this
    work, and the results will be ugly. **No warning is currently given when this happens, so you need to inspect
    your design.** The given radius should be smaller than about half the length of the shortest straight section. If
    several points lie on the same line, the redundant ones are removed.

    :param iterable[indexable] points: a list of points forming the outline of the path to smooth.
    :param float radius: the radius of the circular arcs used to connect the straight segments.
    :param float points_per_degree: the number of points per degree of arc; the default of 1 is usually enough.
    :param bool already_package_format: if True, skip the conversion of the points to package format (used internally
                                        by :class:`SmoothedSegment` to avoid double-conversion).
    :return: four lists with length equal to the number of bends (i.e., two less than the number of given points)
             that contain, for each bend, (1) a list of points in that bend, (2) the bend angle in radians, (3) the
             corner point  from the original list, and (4) the vector offset of the bend arc center relative to the
             corner point.
    :rtype: tuple[list]
    """
    bends = list()
    angles = list()
    corners = list()
    offsets = list()
    if not already_package_format:
        points = to_point_list(points)
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
            arc_angles = (theta + np.pi
                          + np.linspace(-bend_angle / 2, bend_angle / 2,
                                        int(np.ceil(np.degrees(np.abs(bend_angle)) * points_per_degree) + 1)))
            bend = [current + offset + radius * np.array([np.cos(phi), np.sin(phi)]) for phi in arc_angles]
            bends.append(bend)
            angles.append(bend_angle)
            corners.append(current)
            offsets.append(offset)
    return bends, angles, corners, offsets


class SegmentList(list):
    """A list subclass to contain Segments that can be drawn sequentially to form a path.

    Slicing a SegmentList returns a SegmentList so that intermediate points can be computed easily:
    ``segment_list[:4].end`` gives the endpoint of the first four elements joined head to tail.
    """

    def draw(self, cell, origin, flatten=True, individual_keywords=None, **global_keywords):
        """Draw all of the segments contained in this SegmentList into the given cell, connected head to tail, and
        and return the drawn structures.

        The segments are drawn as follows: the origin of the first segment is the given origin, and the origin of
        each subsequent segment is the end of the previous segment.

        :param cell: The cell into which the result is drawn, if not None.
        :type cell: gdspy.Cell or None
        :param indexable origin: The point to use for the origin of the first Segment.
        :param bool flatten: if False, return a list of tuples of the objects returned by the :meth:`draw` methods of
                             each :class:`Segment`, in order; if True, flatten the returned list so that its elements
                             are the structures themselves.
        :param individual_keywords: keys are integer indices and values are dicts of parameters that update the
                                    the :meth:`draw` call for the Segment at that index; use this to override the global
                                    keywords or to pass keywords that not all Segments accept.
        :type individual_keywords: dict or None
        :param global_keywords: keyword arguments passed to every :meth:`Segment.draw`.
        :return: the drawn structures ordered from start to end; see the `flatten` keyword.
        :rtype: list
        """
        if individual_keywords is not None and (min(individual_keywords.keys()) < 0 or
                                                max(individual_keywords.keys()) > len(self)):
            raise ValueError("Index in individual_keywords is outside valid range.")
        drawn = list()
        # It's crucial to avoiding input modification that this also makes a copy.
        point = to_point(origin)
        for index, segment in enumerate(self):
            keywords = global_keywords.copy()
            if individual_keywords is not None:
                keywords.update(individual_keywords.get(index, dict()))
            tuple_of_structures = segment.draw(cell=cell, origin=point, **keywords)
            if flatten:
                try:
                    drawn.extend(tuple_of_structures)
                except TypeError:
                    warnings.warn(f"Appending instead of flattening the non-iterable object returned by Segment at"
                                  f" index {index:d}: {tuple_of_structures!r}")
                    drawn.append(tuple_of_structures)
            else:
                drawn.append(tuple_of_structures)
            # NB: using += produces an error when casting int to float.
            point = point + segment.end
        return drawn

    def __getitem__(self, item):
        """Re-implement this method so that slices return SegmentLists."""
        if isinstance(item, slice):
            return self.__class__(super().__getitem__(item))
        else:
            return super().__getitem__(item)

    @property
    def start(self):
        """The start point of the first element in this SegmentList, assuming its origin is (0, 0)."""
        return self[0].start

    @property
    def end(self):
        """The end point of the last element in this SegmentList, assuming its origin is (0, 0)."""
        return np.sum(np.vstack([element.end for element in self]), axis=0)

    @property
    def span(self):
        """The difference between start and end points: span = end - start, in the vector sense."""
        return self.end - self.start

    @property
    def length(self):
        """The sum of the lengths of the Segments in this SegmentList.

        The calculation sums the `length` properties of all the Segments, and it does **not** check that the Segments
        are all connected head-to-tail.
        """
        return np.sum([element.length for element in self])

    @property
    def points(self):
        """Return a list of lists each containing the points of one Segment in this SegmentList.

        The calculation assumes that the first element starts at (0, 0) and that subsequent elements are placed
        head-to-tail, as when they are drawn, so the points in all lists after the initial one are not the same as the
        points of the corresponding element.

        :return: the as-drawn points of each Segment.
        :rtype: list[list[point]]
        """
        point_lists = list()
        end = to_point((0, 0))
        for element in self:
            point_lists.append([point + end for point in element.points])
            end += element.end
        return point_lists

    @property
    def bounds(self):
        """Return the lower left and upper right points of the smallest rectangle that encloses the points of all
        elements in the SegmentList.

        The calculation assumes that the first element starts at (0, 0) and that subsequent elements are placed
        head-to-tail, as when they are drawn; see :meth:`points`.

        :return: the lower left and upper right points.
        :rtype: tuple[point]
        """
        all_points = [p for element_points in self.points for p in element_points]  # Make a flat list of all points
        xy = np.vstack(all_points).T  # points.shape = (2, num_points)
        return to_point((np.min(xy[0]), np.min(xy[1]))), to_point((np.max(xy[0]), np.max(xy[1])))


class Segment(object):
    """An element in a SegmentList that can draw itself into a cell."""

    def __init__(self, points, round_to=None):
        """The given points are saved as :attr:`_points`, and should generally not be modified.

        :param iterable[indexable] points: the points that form the Segment.
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
    def span(self):
        """The difference between start and end points: span = end - start, in the vector sense."""
        return self.end - self.start

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

    def draw(self, cell, origin, max_points=MAX_POINTS, **keywords):
        """Create and return polygons or other structures and add them to the given cell, if one is specified.

        Subclasses implement this method to draw themselves. They should return an iterable of the drawn structure(
        s), and these structures should not contain more than `max_points` points. Passing `cell=None` should draw
        the structures without adding them to a call, which is useful for temporary structures used for boolean
        operations.

        :param cell: the cell into which this Segment will be drawn, if not None.
        :type cell: gdspy.Cell or None
        :param indexable origin: draw the Segment relative to this point, meaning that point (0, 0) of the Segment is
                                 placed here.
        :return: subclasses should return an iterable of the drawn structure(s).
        :rtype: tuple
        """
        return ()


class SmoothedSegment(Segment):
    """An element in a SegmentList that can draw itself into a cell, with corners smoothed using :func:`smooth`."""

    def __init__(self, outline, radius, points_per_degree, round_to=None):
        """The given outline points are passed to :func:`smooth` and the result is stored in the instance attributes
        :attr:`bends`, :attr:`angles`, :attr:`corners`, and :attr:`offsets`.

        :param iterable[indexable] outline: the outline points, before smoothing.
        :param float radius: the radius of the circular arcs used to connect the straight segments; see :func:`smooth`.
        :param int points_per_degree: the number of points per degree of arc; see :func:`smooth`.
        :param round_to: if not None, the coordinates of each outline point are rounded to this value **before
                         smoothing**; useful for ensuring that all the points in a design lie on a grid larger than the
                         database unit size.
        :type round_to: float, int, or None
        """
        super(SmoothedSegment, self).__init__(points=outline, round_to=round_to)
        self.radius = radius
        self.points_per_degree = points_per_degree
        self.bends, self.angles, self.corners, self.offsets = smooth(self._points, radius, points_per_degree,
                                                                     already_package_format=True)

    @property
    def points(self):
        """The smoothed points (``list[numpy.ndarray]``); the original outline points are ``_points`` (read-only)."""
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
