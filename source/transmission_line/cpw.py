"""This module contains classes for drawing co-planar waveguide transmission lines.

Most of the classes draw structures with **negative polarity**, meaning that structures represent the absence of metal.
"""
from __future__ import absolute_import, division, print_function

import gdspy
import numpy as np

from transmission_line.core import DEFAULT_POINTS_PER_RADIAN, to_point, Segment, SmoothedSegment


class CPW(SmoothedSegment):
    """The negative space of a segment of co-planar waveguide."""

    def __init__(self, outline, width, gap, radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """Instantiating a CPW does not draw it into any cell.

        :param outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float width: the width of the center trace.
        :param float gap: the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.width = width
        self.gap = gap
        if radius is None:
            radius = width / 2 + gap
        super(CPW, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
                                  round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the drawn polygon set, which should contain two polygons.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the object that was drawn into the cell.
        :rtype: gdspy.PolygonSet
        """
        points = [to_point(origin) + point for point in self.points]
        trace_flexpath = gdspy.FlexPath(points=points, width=self.width, max_points=0, gdsii_path=False)
        gap_flexpath = gdspy.FlexPath(points=points, width=self.width + 2 * self.gap, max_points=0, gdsii_path=False)
        polygon_set = gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=0, layer=layer, datatype=datatype)
        cell.add(element=polygon_set)
        return polygon_set


class CPWBlank(SmoothedSegment):
    """Negative co-planar waveguide with the center trace missing, that is, the negative space of the trace and gaps.

     This is useful when the center trace is on a separate layer or has a separate datatype.
     """

    def __init__(self, outline, width, gap, radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """Instantiating a CPWBlank does not draw it into any cell.

        :param outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float width: the width of the center trace.
        :param float gap: the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.width = width
        self.gap = gap
        if radius is None:
            radius = width / 2 + gap
        super(CPWBlank, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
                                       round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the drawn polygon set, .

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the object that was drawn into the cell.
        :rtype: gdspy.PolygonSet
        """
        points = [to_point(origin) + point for point in self.points]
        path_set = gdspy.FlexPath(points=points, width=self.width + 2 * self.gap, layer=layer, datatype=datatype,
                                  max_points=0, gdsii_path=False).to_polygonset()
        cell.add(element=path_set)
        return path_set


class CPWElbowCoupler(SmoothedSegment):
    """Negative co-planar waveguide elbow coupler."""

    def __init__(self, tip_point, elbow_point, joint_point, width, gap, radius=None,
                 points_per_radian=DEFAULT_POINTS_PER_RADIAN,
                 round_to=None):
        """Instantiating a CPWElbowCoupler does not draw it into any cell.

        :param point tip_point: the open end of the coupler; the first point of the segment.
        :param point elbow_point: the point where the coupler turns away from the feedline; the middle point of the
                                  segment.
        :param point joint_point: the point where the coupler joins the rest of the transmission line; the last point of
                                  the segment.
        :param float width: the width of the center trace.
        :param float gap: the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.width = width
        self.gap = gap
        if radius is None:
            radius = width / 2 + gap
        super(CPWElbowCoupler, self).__init__(outline=[tip_point, elbow_point, joint_point], radius=radius,
                                              points_per_radian=points_per_radian, round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0, round_tip=True):
        """Draw this structure into the given cell and return the drawn polygon set, which should contain two polygons.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :param bool round_tip: currently must be True, meaning that the tip that is the start of the path is rounded.
        :return: the path and tip arc that were drawn into the cell.
        :rtype: tuple[gdspy.PolygonSet]
        """
        points = [to_point(origin) + point for point in self.points]
        trace_flexpath = gdspy.FlexPath(points=points, width=self.width, max_points=0, gdsii_path=False)
        gap_flexpath = gdspy.FlexPath(points=points, width=self.width + 2 * self.gap, max_points=0, gdsii_path=False)
        path_set = gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=0, layer=layer, datatype=datatype)
        cell.add(element=path_set)

        if round_tip:
            v = points[0] - points[1]
            theta = np.arctan2(v[1], v[0])
            round_set = gdspy.Round(center=points[0], radius=self.width / 2 + self.gap, inner_radius=self.width / 2,
                                    initial_angle=theta - np.pi / 2, final_angle=theta + np.pi / 2, max_points=0,
                                    layer=layer, datatype=datatype)
            cell.add(element=round_set)
        else:
            raise NotImplementedError("Need to code this up.")

        return path_set, round_set


class CPWElbowCouplerBlank(SmoothedSegment):
    """Negative co-planar waveguide elbow coupler with the center trace missing, that is, the negative space of the
    trace and gaps.

     This is useful when the center trace is on a separate layer or has a separate datatype.
     """

    def __init__(self, tip_point, elbow_point, joint_point, width, gap, radius=None,
                 points_per_radian=DEFAULT_POINTS_PER_RADIAN,
                 round_to=None):
        """Instantiating a CPWElbowCouplerBlank does not draw it into any cell.

        :param point tip_point: the open end of the coupler; the first point of the segment.
        :param point elbow_point: the point where the coupler turns away from the feedline; the middle point of the
                                  segment.
        :param point joint_point: the point where the coupler joins the rest of the transmission line; the last point of
                                  the segment.
        :param float width: the width of the center trace.
        :param float gap: the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.width = width
        self.gap = gap
        if radius is None:
            radius = width / 2 + gap
        super(CPWElbowCouplerBlank, self).__init__(outline=[tip_point, elbow_point, joint_point], radius=radius,
                                                   points_per_radian=points_per_radian, round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0, round_tip=True):
        """Draw this structure into the given cell and return the drawn polygon set, which should contain two polygons.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :param bool round_tip: currently must be True, meaning that the tip that is the start of the path is rounded.
        :return: the path and tip arc that were drawn into the cell.
        :rtype: tuple[gdspy.PolygonSet]
        """
        points = [to_point(origin) + point for point in self.points]
        path_set = gdspy.FlexPath(points=points, width=self.width + 2 * self.gap, layer=layer, datatype=datatype,
                                  max_points=0, gdsii_path=False).to_polygonset()
        cell.add(element=path_set)

        if round_tip:
            v = points[0] - points[1]
            theta = np.arctan2(v[1], v[0])
            round_set = gdspy.Round(center=points[0], radius=self.width / 2 + self.gap, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2, max_points=0, layer=layer, datatype=datatype)
            cell.add(element=round_set)
        else:
            raise NotImplementedError("Need to code this up.")

        return path_set, round_set


class CPWTransition(Segment):
    """Negative transition between two sections of co-planar waveguide."""

    def __init__(self, start_point, end_point, start_width, end_width, start_gap, end_gap, round_to=None):
        """Instantiating a CPWTransition does not draw it into any cell.

        The points of this structure are (start_point, end_point).

        :param point start_point: the start point of the transition, typically the end point of the previous section.
        :param end_point: the end point of the transition, typically the start point of the following section.
        :param start_width: the trace width of the previous section.
        :param end_width: the trace width following section.
        :param start_gap: the gap width of the previous section.
        :param end_gap: the gap width of the following section.
        :param round_to: see :class:`SmoothedSegment`.
        """
        super(CPWTransition, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_width = start_width
        self.start_gap = start_gap
        self.end_width = end_width
        self.end_gap = end_gap

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the two drawn polygons.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the two polygons that were drawn into the cell.
        :rtype: tuple[gdspy.Polygon]
        """
        v = self.end - self.start
        phi = np.arctan2(v[1], v[0])
        rotation = np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
        upper = [(0, self.start_width / 2),
                 (0, self.start_width / 2 + self.start_gap),
                 (self.length, self.end_width / 2 + self.end_gap),
                 (self.length, self.end_width / 2)]
        lower = [(x, -y) for x, y in upper]
        upper_rotated = [np.dot(rotation, to_point(p).T).T for p in upper]
        lower_rotated = [np.dot(rotation, to_point(p).T).T for p in lower]
        upper_rotated_shifted = [to_point(origin) + self.start + p for p in upper_rotated]
        lower_rotated_shifted = [to_point(origin) + self.start + p for p in lower_rotated]
        upper_polygon = gdspy.Polygon(points=upper_rotated_shifted, layer=layer, datatype=datatype)
        lower_polygon = gdspy.Polygon(points=lower_rotated_shifted, layer=layer, datatype=datatype)
        cell.add(element=upper_polygon)
        cell.add(element=lower_polygon)
        return upper_polygon, lower_polygon


class CPWTransitionBlank(Segment):
    """Negative transition between two sections of co-planar waveguide, used when the center trace is a separate
    layer."""

    def __init__(self, start_point, end_point, start_width, end_width, start_gap, end_gap, round_to=None):
        """Instantiating a CPWTransitionBlank does not draw it into any cell.

        The points of this structure are (start_point, end_point).

        :param point start_point: the start point of the transition, typically the end point of the previous section.
        :param end_point: the end point of the transition, typically the start point of the following section.
        :param start_width: the trace width of the previous section.
        :param end_width: the trace width following section.
        :param start_gap: the gap width of the previous section.
        :param end_gap: the gap width of the following section.
        :param round_to: see :class:`SmoothedSegment`.
        """
        super(CPWTransitionBlank, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_width = start_width
        self.start_gap = start_gap
        self.end_width = end_width
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
        points = [(0, self.start_width / 2 + self.start_gap),
                  (self.length, self.end_width / 2 + self.end_gap),
                  (self.length, -self.end_width / 2 - self.end_gap),
                  (0, -self.start_width / 2 - self.start_gap)]
        points_rotated = [np.dot(rotation, to_point(p).T).T for p in points]
        points_rotated_shifted = [to_point(origin) + self.start + p for p in points_rotated]
        polygon = gdspy.Polygon(points=points_rotated_shifted, layer=layer, datatype=datatype)
        cell.add(element=polygon)
        return polygon


# ToDo: split this into separate classes
# ToDo: refactor this so that it's clear that how to mix it with Segment subclasses
class Mesh(object):
    """A mix-in class that allows Segment subclasses that have the same outlines to share mesh code."""

    def path_mesh(self):
        """Return the centers of the mesh elements calculated using the Segment's parameters.

        :return: the centers of the mesh elements.
        :rtype: list[numpy.ndarray]
        """
        mesh_centers = []
        center_to_first_row = self.width / 2 + self.gap + self.mesh_border
        # Mesh the straight sections
        starts = [self.start] + [bend[-1] for bend in self.bends]
        ends = [bend[0] for bend in self.bends] + [self.end]
        for start, end in zip(starts, ends):
            v = end - start
            length = np.linalg.norm(v)
            phi = np.arctan2(v[1], v[0])
            R = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
            num_mesh_columns = int(np.floor(length / self.mesh_spacing))
            if num_mesh_columns == 0:
                continue
            elif num_mesh_columns == 1:
                x = np.array([length / 2])
            else:
                x = np.linspace(self.mesh_spacing / 2, length - self.mesh_spacing / 2, num_mesh_columns)
            y = center_to_first_row + self.mesh_spacing * np.arange(self.num_mesh_rows)
            xx, yy = np.meshgrid(np.concatenate((x, x)), np.concatenate((y, -y)))
            Rxy = np.dot(R, np.vstack((xx.flatten(), yy.flatten())))
            mesh_centers.extend(zip(start[0] + Rxy[0, :], start[1] + Rxy[1, :]))
        # Mesh the curved sections
        for row in range(self.num_mesh_rows):
            center_to_row = center_to_first_row + row * self.mesh_spacing
            for radius in [self.radius - center_to_row, self.radius + center_to_row]:
                if radius < self.mesh_spacing / 2:
                    continue
                for angle, corner, offset in zip(self.angles, self.corners, self.offsets):
                    num_points = int(np.round(radius * np.abs(angle) / self.mesh_spacing))
                    if num_points == 1:
                        max_angle = 0
                    else:
                        max_angle = (1 - 1 / num_points) * angle / 2
                    mesh_centers.extend([corner + offset +
                                         radius * np.array([np.cos(phi), np.sin(phi)]) for phi in
                                         (np.arctan2(-offset[1], -offset[0]) +
                                          np.linspace(-max_angle, max_angle, num_points))])
        return mesh_centers

    def trapezoid_mesh(self):
        """

        :return:
        """
        mesh_centers = []
        v = self.end - self.start
        length = np.linalg.norm(v)
        phi = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
        start_to_first_row = self.start_width / 2 + self.start_gap + self.start_mesh_border
        difference_to_first_row = self.end_width / 2 + self.end_gap + self.end_mesh_border - start_to_first_row
        num_mesh_columns = int(np.floor(length / self.mesh_spacing))
        if num_mesh_columns == 0:
            return []
        elif num_mesh_columns == 1:
            x = np.array([length / 2])
        else:
            x = np.linspace(self.mesh_spacing / 2, length - self.mesh_spacing / 2, num_mesh_columns)
        y = self.mesh_spacing * np.arange(self.num_mesh_rows, dtype=np.float)
        xxp, yyp = np.meshgrid(x, y)  # These correspond to the positive y-values
        y_shift = start_to_first_row + difference_to_first_row * x / length
        yyp += y_shift
        xx = np.concatenate((xxp, xxp))
        yy = np.concatenate((yyp, -yyp))  # The negative y-values are reflected
        Rxy = np.dot(R, np.vstack((xx.flatten(), yy.flatten())))
        mesh_centers.extend(zip(self.start[0] + Rxy[0, :], self.start[1] + Rxy[1, :]))
        return mesh_centers


class CPWMesh(CPW, Mesh):
    """doc me!"""

    def __init__(self, outline, width, gap, mesh_spacing, mesh_border, mesh_radius, num_circle_points, num_mesh_rows,
                 radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """

        :param outline:
        :param width:
        :param gap:
        :param mesh_spacing:
        :param mesh_border:
        :param mesh_radius:
        :param num_circle_points:
        :param num_mesh_rows:
        :param radius:
        :param points_per_radian:
        :param round_to:
        """
        super(CPWMesh, self).__init__(outline=outline, width=width, gap=gap, radius=radius,
                                      points_per_radian=points_per_radian, round_to=round_to)
        self.mesh_spacing = mesh_spacing
        self.mesh_radius = mesh_radius
        self.mesh_border = mesh_border
        self.num_circle_points = num_circle_points
        self.num_mesh_rows = num_mesh_rows
        self.mesh_centers = self.path_mesh()

    def draw(self, cell, origin, positive_layer, negative_layer, result_layer):
        """

        :param cell:
        :param origin:
        :param positive_layer:
        :param negative_layer:
        :param result_layer:
        :return:
        """
        super(CPWMesh, self).draw(cell=cell, origin=origin, positive_layer=positive_layer,
                                  negative_layer=negative_layer, result_layer=result_layer)
        for mesh_center in self.mesh_centers:
            cell.add_circle(origin=origin + mesh_center, radius=self.mesh_radius, layer=result_layer,
                            number_of_points=self.num_circle_points)


class CPWBlankMesh(CPWBlank, Mesh):
    """doc me!"""

    def __init__(self, outline, width, gap, mesh_spacing, mesh_border, mesh_radius, num_circle_points, num_mesh_rows,
                 radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """

        :param outline:
        :param width:
        :param gap:
        :param mesh_spacing:
        :param mesh_border:
        :param mesh_radius:
        :param num_circle_points:
        :param num_mesh_rows:
        :param radius:
        :param points_per_radian:
        :param round_to:
        """
        super(CPWBlankMesh, self).__init__(outline=outline, width=width, gap=gap, radius=radius,
                                           points_per_radian=points_per_radian, round_to=round_to)
        self.mesh_spacing = mesh_spacing
        self.mesh_radius = mesh_radius
        self.mesh_border = mesh_border
        self.num_circle_points = num_circle_points
        self.num_mesh_rows = num_mesh_rows
        self.mesh_centers = self.path_mesh()

    def draw(self, cell, origin, positive_layer, negative_layer, result_layer):
        """

        :param cell:
        :param origin:
        :param positive_layer:
        :param negative_layer:
        :param result_layer:
        :return:
        """
        super(CPWBlankMesh, self).draw(cell=cell, origin=origin, positive_layer=positive_layer,
                                       negative_layer=negative_layer, result_layer=result_layer)
        for mesh_center in self.mesh_centers:
            cell.add_circle(origin=origin + mesh_center, radius=self.mesh_radius, layer=result_layer,
                            number_of_points=self.num_circle_points)


class CPWElbowCouplerMesh(CPWElbowCoupler):
    """doc me!"""
    pass


class CPWElbowCouplerBlankMesh(CPWElbowCouplerBlank):
    """doc me!"""
    pass


class CPWTransitionMesh(CPWTransition, Mesh):
    """doc me!"""

    def __init__(self, start_point, end_point, start_width, end_width, start_gap, end_gap, mesh_spacing,
                 start_mesh_border, end_mesh_border, mesh_radius, num_circle_points, num_mesh_rows, round_to=None):
        """

        :param start_point:
        :param end_point:
        :param start_width:
        :param end_width:
        :param start_gap:
        :param end_gap:
        :param mesh_spacing:
        :param start_mesh_border:
        :param end_mesh_border:
        :param mesh_radius:
        :param num_circle_points:
        :param num_mesh_rows:
        :param round_to:
        """
        super(CPWTransitionMesh, self).__init__(start_point=start_point, end_point=end_point, start_width=start_width,
                                                end_width=end_width, start_gap=start_gap, end_gap=end_gap,
                                                round_to=round_to)
        self.mesh_spacing = mesh_spacing
        self.start_mesh_border = start_mesh_border
        self.end_mesh_border = end_mesh_border
        self.mesh_radius = mesh_radius
        self.num_circle_points = num_circle_points
        self.num_mesh_rows = num_mesh_rows
        self.mesh_centers = self.trapezoid_mesh()

    def draw(self, cell, origin, positive_layer, negative_layer, result_layer):
        """

        :param cell:
        :param origin:
        :param positive_layer:
        :param negative_layer:
        :param result_layer:
        :return:
        """
        super(CPWTransitionMesh, self).draw(cell=cell, origin=origin, positive_layer=positive_layer,
                                            negative_layer=negative_layer, result_layer=result_layer)
        for mesh_center in self.mesh_centers:
            cell.add_circle(origin=origin + mesh_center, radius=self.mesh_radius, layer=result_layer,
                            number_of_points=self.num_circle_points)


class CPWTransitionBlankMesh(CPWTransitionBlank, Mesh):
    """doc me!"""

    def __init__(self, start_point, end_point, start_width, end_width, start_gap, end_gap, mesh_spacing,
                 start_mesh_border, end_mesh_border, mesh_radius, num_circle_points, num_mesh_rows, round_to=None):
        """

        :param start_point:
        :param end_point:
        :param start_width:
        :param end_width:
        :param start_gap:
        :param end_gap:
        :param mesh_spacing:
        :param start_mesh_border:
        :param end_mesh_border:
        :param mesh_radius:
        :param num_circle_points:
        :param num_mesh_rows:
        :param round_to:
        """
        super(CPWTransitionBlankMesh, self).__init__(start_point=start_point, end_point=end_point,
                                                     start_width=start_width, end_width=end_width,
                                                     start_gap=start_gap, end_gap=end_gap, round_to=round_to)
        self.mesh_spacing = mesh_spacing
        self.start_mesh_border = start_mesh_border
        self.end_mesh_border = end_mesh_border
        self.mesh_radius = mesh_radius
        self.num_circle_points = num_circle_points
        self.num_mesh_rows = num_mesh_rows
        self.mesh_centers = self.trapezoid_mesh()

    def draw(self, cell, origin, positive_layer, negative_layer, result_layer):
        """

        :param cell:
        :param origin:
        :param positive_layer:
        :param negative_layer:
        :param result_layer:
        :return:
        """
        super(CPWTransitionBlankMesh, self).draw(cell=cell, origin=origin, positive_layer=positive_layer,
                                                 negative_layer=negative_layer, result_layer=result_layer)
        for mesh_center in self.mesh_centers:
            cell.add_circle(origin=origin + mesh_center, radius=self.mesh_radius, layer=result_layer,
                            number_of_points=self.num_circle_points)
