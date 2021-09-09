"""This module contains equations for calculating properties of co-planar waveguide transmission lines, such as their
inductance and capacitance per unit length, and classes for drawing them as GDSII structures.
"""
from __future__ import absolute_import, division, print_function

import gdspy
import numpy as np
from scipy.constants import c, epsilon_0, mu_0, pi
from scipy.special import ellipk

from transmission_line.transmission_line import (POINTS_PER_DEGREE, MAX_POINTS, to_point, AbstractTransmissionLine,
                                                 Segment, SmoothedSegment)


def half_capacitance_per_unit_length_zero_thickness(trace, gap, dielectric_constant):
    """Return the capacitance per unit length of a zero-thickness CPW due to one semi-infinite space ending in the CPW
    plane, ignoring the capacitance due to the other semi-infinite space.

    The result depends only on the ratios of the lengths, so they can be specified in any units as long as they are all
    the same; the capacitance is returned in F/m.

    :param float trace: the width of the center trace in the same units as gap.
    :param float gap: the width of the gaps in the same units as trace.
    :param float dielectric_constant: the relative dielectric constant of the semi-infinite space.
    :return: the capacitance in farads per meter.
    :rtype: float
    """
    k = trace / (trace + 2 * gap)
    return 2 * epsilon_0 * dielectric_constant * ellipk(k ** 2) / ellipk(1 - k ** 2)


def capacitance_per_unit_length_zero_thickness(trace, gap, substrate_dielectric_constant, other_dielectric_constant=1):
    """Return the capacitance per unit length of a zero-thickness CPW in the plane separating two semi-infinite spaces
    with the given dielectric constants.

    The result depends only on the ratios of the lengths, so they can be specified in any units as long as they are all
    the same; the capacitance is returned in F/m.

    :param float trace: the width of the center trace in the same units as gap.
    :param float gap: the width of the gaps in the same units as trace.
    :param float substrate_dielectric_constant: the relative dielectric constant of the substrate, one semi-infinite
                                                space.
    :param float other_dielectric_constant: the relative dielectric constant of the other semi-infinite space; by
                                            default this equals 1, corresponding to vacuum.
    :return: the capacitance in farads per meter.
    :rtype: float
    """
    k = trace / (trace + 2 * gap)
    effective_dielectric_constant = (substrate_dielectric_constant + other_dielectric_constant) / 2
    return 4 * epsilon_0 * effective_dielectric_constant * ellipk(k ** 2) / ellipk(1 - k ** 2)


def geometric_inductance_per_unit_length_zero_thickness(trace, gap):
    """Return the geometric inductance per unit length of a zero-thickness CPW with the given geometry.

    The result depends only on the ratios of the lengths, so they can be specified in any units as long as they are all
    the same; the inductance is returned in H/m. The surrounding materials are assumed to have relative permeability
    equal to 1.

    :param float trace: the width of the center trace in the same units as gap.
    :param float gap: the width of the gaps in the same units as trace.
    :return: the inductance in henries per meter.
    :rtype: float
    """
    k = trace / (trace + 2 * gap)
    return (mu_0 / 4) * ellipk(1 - k ** 2) / ellipk(k ** 2)


# ToDo: add comments explaining these with links to thesis
# Equations from Jiansong Gao thesis (JG)

def jg_equation_327_u1(a, b, t):
    """JG Equation 3.27"""
    d = 2 * t / pi
    return a + d / 2 * (1 + 3 * np.log(2) - np.log(d / a) + np.log((b - a) / (a + b)))


def jg_equation_327_u2(a, b, t):
    """JG Equation 3.27"""
    d = 2 * t / pi
    return b + d / 2 * (-1 - 3 * np.log(2) + np.log(d / b) - np.log((b - a) / (a + b)))


def half_capacitance_per_unit_length_finite_thickness(trace, gap, thickness, dielectric_constant):
    """Return the capacitance per unit length of a finite-thickness CPW due to one semi-infinite space ending in the CPW
    plane, ignoring the capacitance due to the other semi-infinite space.

    The result depends only on ratios of the lengths, so they can be specified in any units as long as they are all the
    same; the capacitance is returned in F/m.

    :param float trace: the width of the center trace in the same units as the other lengths.
    :param float gap: the width of the gaps in the same units as the other lengths.
    :param float thickness: the thickness of the metal that is inside the semi-infinite space, in the same units as the
                            other lengths.
    :param float dielectric_constant: the relative dielectric constant of the material filling the semi-infinite space.
    :return: the capacitance per unit length in farads per meter due to fields in this semi-infinite space.
    :rtype: float
    """
    kt = (jg_equation_327_u1(a=trace / 2, b=trace / 2 + gap, t=thickness / 2) /
          jg_equation_327_u2(a=trace / 2, b=trace / 2 + gap, t=thickness / 2))
    return 2 * epsilon_0 * dielectric_constant * ellipk(kt ** 2) / ellipk(1 - kt ** 2)


def capacitance_per_unit_length_finite_thickness(trace, gap, thickness, substrate_dielectric_constant):
    """Return the capacitance per unit length of a finite-thickness CPW on a substrate with the given dielectric
    constant, with the remaining space assumed to be filled by vacuum.

    See JG Equation 3.30.

    The capacitance per unit length is given by the sum of
    the half-capacitance per unit length of a finite-thickness CPW in vacuum;
    the half-capacitance per unit length of a zero-thickness CPW on the substrate.

    :param float trace: the width of the center trace.
    :param float gap: the width of the gap.
    :param float thickness: the thickness of the metal.
    :param float dielectric_constant: the relative dielectric constant of the semi-infinite space.
    :return: the half-capacitance in farads per meter.
    :rtype: float
    """
    return (half_capacitance_per_unit_length_zero_thickness(trace=trace, gap=gap,
                                                            dielectric_constant=substrate_dielectric_constant) +
            half_capacitance_per_unit_length_finite_thickness(trace=trace, gap=gap, thickness=thickness,
                                                              dielectric_constant=1))


def geometric_inductance_per_unit_length_finite_thickness(trace, gap, thickness):
    """Return the geometric inductance per unit length of a finite-thickness CPW.

    See JG Equation 3.31

    The result depends only on ratios of the lengths, so they can be specified in any units as long as they are all the
    same; the inductance is returned in H/m. The equation calculates the parallel inductance of two half-CPWs each with
    half the given thickness.

    :param float trace: the width of the center trace in the same units as the other lengths.
    :param float gap: the width of the gaps in the same units as the other lengths.
    :param float thickness: the total thickness of the metal in the same units as the other lengths.
    :return: the inductance per unit length in farads per meter.
    :rtype: float
    """
    kt = (jg_equation_327_u1(a=trace / 2, b=trace / 2 + gap, t=thickness / 2) /
          jg_equation_327_u2(a=trace / 2, b=trace / 2 + gap, t=thickness / 2))
    return (mu_0 / 4) * ellipk(1 - kt ** 2) / ellipk(kt ** 2)


# Equations from Rami Barends thesis (RB)

def geometry_factor_trace(trace, gap, thickness):
    """Return the kinetic inductance geometry factor for the central conducting trace of a CPW.

    If the kinetic inductance of the central trace is L_k, its kinetic inductance contribution per unit length is
      L = g_c L_k,
    where g_c is the geometry factor returned by this function.

    The trace, gap, and thickness must all be given in the same units, and the returned value will be in the inverse of
    these units.

    :param float trace: the width of the center trace.
    :param float gap: the width of the gaps.
    :param float thickness: the thickness of the metal.
    :return: the geometry factor in the inverse of the length unit.
    :rtype: float
    """
    k = trace / (trace + 2 * gap)
    return (1 / (4 * trace * (1 - k ** 2) * ellipk(k ** 2) ** 2)
            * (pi + np.log(4 * pi * trace / thickness) - k * np.log((1 + k) / (1 - k))))


def geometry_factor_ground(trace, gap, thickness):
    """Return the kinetic inductance geometry factor for the ground planes of a CPW.

    If the kinetic inductance of the ground planes is L_k, their kinetic inductance contribution per unit length is
      L = g_g L_k,
    where g_g is the geometry factor returned by this function.

    The trace, gap, and thickness must all be given in the same units, and the returned value will be in the inverse of
    these units.

    :param float trace: the width of the center trace.
    :param float gap: the width of the gaps.
    :param float thickness: the thickness of the metal.
    :return: the geometry factor in the inverse of the length unit.
    :rtype: float
    """
    k = trace / (trace + 2 * gap)
    return (k / (4 * trace * (1 - k ** 2) * ellipk(k ** 2) ** 2)
            * (pi + np.log(4 * pi * (trace + 2 * gap) / thickness) - (1 / k) * np.log((1 + k) / (1 - k))))


class AbstractCPW(AbstractTransmissionLine):
    """An abstract co-planar waveguide on a substrate, with transverse dimensions but no length.

    Use this to calculate quantities like characteristic impedance or phase velocity. The CPW classes below inherit from
    this class in order to calculate their transmission line properties.
    """

    def __init__(self, trace, gap, thickness=None, substrate_dielectric_constant=1, other_dielectric_constant=1,
                 trace_kinetic_inductance=0, ground_kinetic_inductance=0):
        """The default values correspond to a zero-thickness CPW surrounded by vacuum with no kinetic inductance.

        If thickness is None, the default, then the zero-thickness equations are used for the capacitance and the
        geometric inductance, also, because the geometry factors cannot be calculated, the total inductance per unit
        length equals the geometric inductance per unit length and all properties involving kinetic inductance will
        raise ValueError.

        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps.
        :param float thickness: the thickness of the metal, default None (kinetic inductance not calculable).
        :param float substrate_dielectric_constant: the dielectric constant of the substrate, default 1 (vacuum).
        :param float other_dielectric_constant: the dielectric constant of the non-substrate space, default 1 (vacuum).
        :param float trace_kinetic_inductance: the kinetic inductance of the center trace metal in henries.
        :param float ground_kinetic_inductance: the kinetic inductance of the ground plane metal in henries.
        """
        self.trace = trace
        self.gap = gap
        self.thickness = thickness
        self.substrate_dielectric_constant = substrate_dielectric_constant
        self.other_dielectric_constant = other_dielectric_constant
        self.trace_kinetic_inductance = trace_kinetic_inductance
        self.ground_kinetic_inductance = ground_kinetic_inductance

    @property
    def capacitance_per_unit_length(self):
        """Return the capacitance per unit length in F/m; see :func:`capacitance_per_unit_length`."""
        if self.thickness is None:
            return capacitance_per_unit_length_zero_thickness(
                trace=self.trace, gap=self.gap, substrate_dielectric_constant=self.substrate_dielectric_constant)
        else:
            return capacitance_per_unit_length_finite_thickness(
                trace=self.trace, gap=self.gap, thickness=self.thickness,
                substrate_dielectric_constant=self.substrate_dielectric_constant)

    @property
    def geometric_inductance_per_unit_length(self):
        """Return the geometric inductance per unit length in H/m; see :func:`geometric_inductance_per_unit_length`."""
        if self.thickness is None:
            return geometric_inductance_per_unit_length_zero_thickness(trace=self.trace, gap=self.gap)
        else:
            return geometric_inductance_per_unit_length_finite_thickness(trace=self.trace, gap=self.gap,
                                                                         thickness=self.thickness)

    @property
    def geometry_factor_trace(self):
        """Return the geometry factor for the center trace; see :func:`geometry_factor_trace`."""
        return geometry_factor_trace(trace=self.trace, gap=self.gap, thickness=self.thickness)

    @property
    def geometry_factor_ground(self):
        """Return the geometry factor for the ground planes; see :func:`geometry_factor_ground`."""
        return geometry_factor_ground(trace=self.trace, gap=self.gap, thickness=self.thickness)

    @property
    def kinetic_inductance_per_unit_length_trace(self):
        """Return the kinetic inductance per unit length due to the center trace; see
        :func:`kinetic_inductance_per_unit_length_trace`.
        """
        return self.geometry_factor_trace * self.trace_kinetic_inductance

    @property
    def kinetic_inductance_per_unit_length_ground(self):
        """Return the kinetic inductance per unit length due to the ground planes; see
        :func:`kinetic_inductance_per_unit_length_ground`.
        """
        return self.geometry_factor_ground * self.ground_kinetic_inductance

    @property
    def kinetic_inductance_per_unit_length(self):
        """Return the total (center trace + ground plane) kinetic inductance."""
        return self.kinetic_inductance_per_unit_length_trace + self.kinetic_inductance_per_unit_length_ground

    @property
    def inductance_per_unit_length(self):
        """Return the total (geometric + kinetic, if thickness was given) inductance per unit length."""
        if self.thickness is None:
            return self.geometric_inductance_per_unit_length
        else:
            return self.geometric_inductance_per_unit_length + self.kinetic_inductance_per_unit_length


# ToDo: determine how to handle the multiple inheritance for AbstractCPW and SmoothedSegment


class CPW(SmoothedSegment):
    """A segment of co-planar waveguide.

    Boolean keywords control which structures are drawn from among the trace, the gaps, and the ground planes. Thus,
    structures may represent either metal or its absence.
    """

    def __init__(self, outline, trace, gap, ground=None, radius=None, points_per_degree=POINTS_PER_DEGREE,
                 round_to=None):
        """Instantiate without drawing any structures.

        :param outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float trace: the width of the center trace metal
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param ground: the width of the ground plane metal, which must be specified if these are to be drawn, but can be
                       omitted (default) if only the negative space, the gaps, is to be drawn.
        :type ground: float or None
        :param radius: the default bend radius is the sum of the trace and gap widths, which avoids a sharp interior
                       corner; see :func:`smooth`.
        :type radius: float or None
        :param float points_per_degree: see :func:`smooth`.
        :param round_to: if not None, outline points are rounded to this value; see :class:`SmoothedSegment`.
        :type round_to: float or None
        """
        self.trace = trace
        self.gap = gap
        self.ground = ground
        if radius is None:
            radius = trace + gap
        super(CPW, self).__init__(outline=outline, radius=radius, points_per_degree=points_per_degree,
                                  round_to=round_to)

    def draw(self, cell, origin, layer=0, datatype=0, draw_trace=False, draw_gap=True, draw_ground=False,
             trace_ends='flush', gap_ends='flush', ground_ends='flush', max_points=MAX_POINTS):
        """Draw the specified structure(s) into the given cell (if not None) and return a tuple of polygons.

        The boolean keywords `draw_trace`, `draw_gap`, and `draw_ground` can be used to draw any combination of
        the three possible structures. For example, the default values of `draw_trace=False`, `draw_gap=True`,
        `draw_ground=False` draws only the CPW gaps, which represent absence of metal. To draw instead the
        structures that represent metal in a CPW with specified ground planes, instantiate with `draw_ground` not
        equal to None and draw using `draw_trace=True`, `draw_gap=False`, `draw_ground=True`.

        In order to draw any 'positive' CPW structures, both start and end ground widths must be given.

        :param cell: the cell into which to draw the structure, if not None.
        :type cell: gdspy.Cell or None
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :param bool draw_trace: if True, draw the center trace.
        :param bool draw_gap: if True, draw the gaps.
        :param bool draw_ground: if True, draw the ground planes; the instance must have been created with `ground`
                                 equal to a number, not the default of None.
        :param int max_points: drawn polygons with more than this many points are fractured.
        :return: the drawn structures, regardless of whether they were added to a cell.
        :rtype: tuple[gdspy.PolygonSet]
        """
        points = [to_point(origin) + point for point in self.points]
        if draw_trace and draw_gap and draw_ground:
            flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                      max_points=max_points, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(),)
        elif draw_trace and draw_gap:
            flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap,
                                      max_points=max_points, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(),)
        elif draw_gap and draw_ground:
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground)
            polygons = (gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=max_points,
                                      layer=layer, datatype=datatype),)
        elif draw_trace and draw_ground:  # Positive CPW
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace)
            gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground)
            negative_polygons = gdspy.boolean(gap_flexpath, trace_flexpath, 'not')
            polygons = (gdspy.boolean(ground_flexpath, negative_polygons, 'not', max_points=max_points,
                                      layer=layer, datatype=datatype),)
        elif draw_trace:
            flexpath = gdspy.FlexPath(points=points, width=self.trace, max_points=max_points, layer=layer,
                                      datatype=datatype)
            polygons = (flexpath.to_polygonset(),)
        elif draw_gap:  # Negative CPW, the default
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace)
            gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap)
            polygons = (gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=max_points,
                                      layer=layer, datatype=datatype),)
        elif draw_ground:
            trace_and_gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground)
            polygons = (gdspy.boolean(ground_flexpath, trace_and_gap_flexpath, 'not', max_points=max_points,
                                      layer=layer, datatype=datatype),)
        else:  # Draw nothing
            polygons = ()
        if cell is not None:
            for polygon in polygons:
                cell.add(element=polygon)
        return polygons


# ToDo: include strip or not?
class CPWElbowCoupler(SmoothedSegment):
    """A CPW elbow coupler."""

    def __init__(self, open_point, elbow_point, joint_point, trace, gap, ground=None, strip=None, open_at_start=True,
                 radius=None, points_per_degree=POINTS_PER_DEGREE, round_to=None):
        """Instantiate without drawing in any cell.

        If `open_at_start` is True (default), the structure should be used as the initial element of a SegmentList and
        its outline points are ``[open_point, elbow_point, joint_point]``.

        If `open_at_start` is False, the structure should be used as the final element of a SegmentList and its outline
        points are ``[joint_point, elbow_point, open_point]``. If ``joint_point = (0, 0)``, it will be connected to the
        previous segment.

        This is a subclass of :class:`SmoothedSegment`, so the elbow is rounded. Electromagnetic simulations show that
        the bend contributes little to the coupling, so a good approximation to the effective coupler length may be the
        outline length minus the bend radius.

        :param point open_point: the open end of the coupler.
        :param point elbow_point: the point where the coupler turns away from the feedline; the middle point of the
                                  segment.
        :param point joint_point: the point where the coupler joins the rest of the transmission line.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param ground: the width of the ground plane metal, which must be specified if these are to be drawn, but can be
                       omitted (default) if only the negative space, the gaps, is to be drawn.
        :type ground: float or None
        :param strip: not yet implemented; this is intended to be the width of the ground plane metal between the gaps
                      of the coupler CPW and the feedline CPW, and it will be used only when the ground is drawn, as for
                      positive CPW.
        :type strip: float or None
        :param bool open_at_start: if True (default), the open is at the start point and this structure should be placed
                                   first in a SegmentList; if False, it is at the end point and this structure should be
                                   placed last in a SegmentList.
        :param radius: the radius of the elbow bend (see :func:`smooth`); if None, the default is the sum of the trace
                       and gap widths.
        :type radius: float or None
        :param float points_per_degree: see :func:`smooth`.
        :param round_to: if not None, outline points are rounded to this value; see :class:`SmoothedSegment`.
        :type round_to: float or None
        """
        self.trace = trace
        self.gap = gap
        self.ground = ground
        if radius is None:
            radius = trace + gap
        self.open_at_start = bool(open_at_start)
        if self.open_at_start:
            outline = [open_point, elbow_point, joint_point]
        else:
            outline = [joint_point, elbow_point, open_point]
        super(CPWElbowCoupler, self).__init__(outline=outline, radius=radius, points_per_degree=points_per_degree,
                                              round_to=round_to)

    def draw(self, cell, origin, layer=0, datatype=0, draw_trace=False, draw_gap=True, draw_ground=False,
             overlap=0, ground_extension=None, max_points=MAX_POINTS):
        """Draw the specified structure(s) into the given cell (if not None) and return a tuple of polygons.

        The boolean keywords `draw_trace`, `draw_gap`, and `draw_ground` can be used to draw any combination of the
        three possible structures. For example, the default values of `draw_trace=False`, `draw_gap=True`,
        `draw_ground=False` draws only the CPW gaps, which represent absence of metal. To draw instead the structures
        that represent metal in a 'positive' CPW, with specified ground planes, instantiate with `ground` equal to a
        number then call this method with `draw_trace=True`, `draw_gap=False`, `draw_ground=True`.

        Structures that are adjacent, such as the trace and gaps, are not returned as separate polygons but rather as
        single polygons that are the union of the individual structures.

        :param cell: the cell into which to draw the structure, if not None.
        :type cell: gdspy.Cell or None
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :param bool draw_trace: if True, draw the center trace.
        :param bool draw_gap: if True, draw the gaps.
        :param bool draw_ground: if True, draw the ground planes; the instance must have been created with `ground`
                                 equal to a number, not the default of None.
        :param float overlap: all drawn structures are extended this distance at the end opposite the open.
        :param ground_extension: if None, the ground plane is extended at the open end by a distance equal to half the
                                 trace plus the gap; in this case, if the gap is also drawn, its rounded arc will touch
                                 the end of the ground plane.
        :type ground_extension: float or None
        :return: the drawn structures, regardless of whether they were added to a cell.
        :rtype: tuple[gdspy.PolygonSet]
        """
        points = [to_point(origin) + point for point in self.points]
        if ground_extension is None:
            ground_extension = self.trace / 2 + self.gap
        if self.open_at_start:
            ultimate = points[0]
            penultimate = points[1]
            trace_ends = (0, overlap)
            gap_ends = (0, overlap)
            ground_ends = (ground_extension, overlap)
        else:
            ultimate = points[-1]
            penultimate = points[-2]
            trace_ends = (overlap, 0)
            gap_ends = (overlap, 0)
            ground_ends = (overlap, ground_extension)
        v = ultimate - penultimate  # This vector points toward the open end
        theta = np.arctan2(v[1], v[0])
        if draw_trace and draw_gap and draw_ground:  # No rounded end
            flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                      ends=ground_ends, max_points=max_points, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(),)
        elif draw_trace and draw_gap:
            flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap,
                                      ends=gap_ends, max_points=max_points, layer=layer, datatype=datatype)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(), cap)
        elif draw_gap and draw_ground:
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                             ends=ground_ends)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2)
            polygons = (gdspy.boolean(ground_flexpath, [trace_flexpath.to_polygonset(), cap], 'not',
                                      max_points=max_points, layer=layer, datatype=datatype),)
        elif draw_trace and draw_ground:  # Positive CPW
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends)
            trace_cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2)
            gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, ends=gap_ends)
            gap_cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                                  final_angle=theta + np.pi / 2)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                             ends=ground_ends)
            negative_polygons = gdspy.boolean([gap_flexpath.to_polygonset(), gap_cap],
                                              [trace_flexpath.to_polygonset(), trace_cap], 'not')
            polygons = (gdspy.boolean(ground_flexpath, negative_polygons, 'not', max_points=max_points,
                                      layer=layer, datatype=datatype),)
        elif draw_trace:
            flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends, max_points=max_points,
                                      layer=layer, datatype=datatype)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(), cap)
        elif draw_gap:  # Negative CPW, the default
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends)
            trace_cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2)
            gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, ends=gap_ends)
            gap_cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                                  final_angle=theta + np.pi / 2)
            polygons = (gdspy.boolean([gap_flexpath.to_polygonset(), gap_cap],
                                      [trace_flexpath.to_polygonset(), trace_cap], 'not',
                                      max_points=max_points, layer=layer, datatype=datatype),)
        elif draw_ground:
            trace_and_gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, ends=gap_ends)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                             ends=ground_ends)
            polygons = (gdspy.boolean(ground_flexpath, [trace_and_gap_flexpath.to_polygonset(), cap], 'not',
                                      max_points=max_points, layer=layer, datatype=datatype),)
        else:  # Draw nothing
            polygons = ()
        if cell is not None:
            for polygon in polygons:
                cell.add(element=polygon)
        return polygons


class CPWRoundedOpen(Segment):
    """A CPW that terminates in an open with a rounded end cap.

    It can be used as either the initial or the final element of a :class:`SegmentList`.
    """

    def __init__(self, joint_point, open_point, trace, gap, ground=None, round_to=None, open_at_start=False):
        """Instantiate without drawing in any cell.

        If `open_at_start` is False (default), the points of this Segment are ``(joint_point, open_point)``. In this
        case, this structure will be connected with the preceding one if it is the final element in a SegmentList
        and if ``joint_point=(0, 0)``.

        If `open_at_start` is True, the points of this Segment are ``(open_point, joint_point)``. In this case, this
        structure can be the first element a SegmentList.

        Since there are no bends, it is not smoothed.

        :param point start_point: the start of the segment.
        :param point end_point: the end of the segment.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param ground: the width of the ground plane metal, which must be specified if these are to be drawn, but can be
                       omitted (default) if only the negative space, the gaps, is to be drawn.
        :type ground: float or None
        :param round_to: if not None, start and end points are rounded to this value; see :class:`Segment`.
        :type round_to: float or None
        :param bool open_at_start: if False (default), the open is at the end point and this structure should be placed
                                   last in a SegmentList; if True, it is at the start point and this structure should be
                                   placed first in a SegmentList.
        """
        self.trace = trace
        self.gap = gap
        self.ground = ground
        self.open_at_start = bool(open_at_start)
        if self.open_at_start:
            points = [open_point, joint_point]
        else:
            points = [joint_point, open_point]
        super(CPWRoundedOpen, self).__init__(points=points, round_to=round_to)

    def draw(self, cell, origin, layer=0, datatype=0, draw_trace=False, draw_gap=True, draw_ground=False,
             overlap=0, ground_extension=None, max_points=MAX_POINTS):
        """Draw the specified structure(s) into the given cell (if not None) and return a tuple of polygons.

        The boolean keywords `draw_trace`, `draw_gap`, and `draw_ground` can be used to draw any combination of the
        three possible structures. For example, the default values of `draw_trace=False`, `draw_gap=True`,
        `draw_ground=False` draws only the CPW gaps, which represent absence of metal. To draw instead the structures
        that represent metal in a 'positive' CPW, with specified ground planes, instantiate with `ground` equal to a
        number then call this method with `draw_trace=True`, `draw_gap=False`, `draw_ground=True`.

        Structures that are adjacent, such as the trace and gaps, are not returned as separate polygons but rather as
        single polygons that are the union of the individual structures.

        :param cell: the cell into which to draw the structure, if not None.
        :type cell: gdspy.Cell or None
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :param bool draw_trace: if True, draw the center trace.
        :param bool draw_gap: if True, draw the gaps.
        :param bool draw_ground: if True, draw the ground planes; the instance must have been created with `ground`
                                 equal to a number, not the default of None.
        :param float overlap: all drawn structures are extended this distance at the end opposite the open.
        :param ground_extension: if None, the ground plane is extended at the open end by a distance equal to half the
                                 trace plus the gap; in this case, if the gap is also drawn, its rounded arc will touch
                                 the end of the ground plane.
        :type ground_extension: float or None
        :return: the drawn structures, regardless of whether they were added to a cell.
        :rtype: tuple[gdspy.PolygonSet]
        """
        points = [to_point(origin) + point for point in self.points]
        if ground_extension is None:
            ground_extension = self.trace / 2 + self.gap
        if self.open_at_start:
            ultimate, penultimate = points
            trace_ends = (0, overlap)
            gap_ends = (0, overlap)
            ground_ends = (ground_extension, overlap)
        else:
            penultimate, ultimate = points
            trace_ends = (overlap, 0)
            gap_ends = (overlap, 0)
            ground_ends = (overlap, ground_extension)
        v = ultimate - penultimate  # This vector points toward the open end
        theta = np.arctan2(v[1], v[0])
        if draw_trace and draw_gap and draw_ground:  # No rounded end
            flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                      ends=ground_ends, max_points=max_points, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(),)
        elif draw_trace and draw_gap:
            flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap,
                                      ends=gap_ends, max_points=max_points, layer=layer, datatype=datatype)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(), cap)
        elif draw_gap and draw_ground:
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                             ends=ground_ends)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2)
            polygons = (gdspy.boolean(ground_flexpath, [trace_flexpath.to_polygonset(), cap], 'not',
                                      max_points=max_points, layer=layer, datatype=datatype),)
        elif draw_trace and draw_ground:  # Positive CPW
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends)
            trace_cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2)
            gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, ends=gap_ends)
            gap_cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                                  final_angle=theta + np.pi / 2)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                             ends=ground_ends)
            negative_polygons = gdspy.boolean([gap_flexpath.to_polygonset(), gap_cap],
                                              [trace_flexpath.to_polygonset(), trace_cap], 'not')
            polygons = (gdspy.boolean(ground_flexpath, negative_polygons, 'not', max_points=max_points,
                                      layer=layer, datatype=datatype),)
        elif draw_trace:
            flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends, max_points=max_points,
                                      layer=layer, datatype=datatype)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2, layer=layer, datatype=datatype)
            polygons = (flexpath.to_polygonset(), cap)
        elif draw_gap:  # Negative CPW, the default
            trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, ends=trace_ends)
            trace_cap = gdspy.Round(center=ultimate, radius=self.trace / 2, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2)
            gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, ends=gap_ends)
            gap_cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                                  final_angle=theta + np.pi / 2)
            polygons = (gdspy.boolean([gap_flexpath.to_polygonset(), gap_cap],
                                      [trace_flexpath.to_polygonset(), trace_cap], 'not',
                                      max_points=max_points, layer=layer, datatype=datatype),)
        elif draw_ground:
            trace_and_gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, ends=gap_ends)
            cap = gdspy.Round(center=ultimate, radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                              final_angle=theta + np.pi / 2)
            ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground,
                                             ends=ground_ends)
            polygons = (gdspy.boolean(ground_flexpath, [trace_and_gap_flexpath.to_polygonset(), cap], 'not',
                                      max_points=max_points, layer=layer, datatype=datatype),)
        else:  # Draw nothing
            polygons = ()
        if cell is not None:
            for polygon in polygons:
                cell.add(element=polygon)
        return polygons


class CPWTransition(Segment):
    """Transition between two sections of co-planar waveguide.

    The points of this structure are [start_point, end_point].
    """

    def __init__(self, start_point, end_point, start_trace, end_trace, start_gap, end_gap, start_ground=None,
                 end_ground=None, round_to=None):
        """Instantiate without drawing any structures.

        In order to draw any 'positive' CPW structures, both start and end ground widths must be given. This structure
        does not support overlaps, which should be drawn using the adjacent structures.

        :param point start_point: the start point of the transition; typically (0, 0), so that it will be connected to
                                  the previous Segment.
        :param point end_point: the end point of the transition.
        :param float start_trace: the trace width of the previous Segment.
        :param float end_trace: the trace width of the following Segment.
        :param float start_gap: the gap width of the previous Segment.
        :param float end_gap: the gap width of the following Segment.
        :param start_ground: the ground width of the previous Segment; must be specified to draw grounds.
        :type start_ground: float or None
        :param end_ground: the ground width of the following Segment; must be specified to draw grounds.
        :type end_ground: float or None
        :param round_to: if not None, start and end points are rounded to this value; see :class:`Segment`.
        :type round_to: float or None
        """
        super(CPWTransition, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_trace = start_trace
        self.end_trace = end_trace
        self.start_gap = start_gap
        self.end_gap = end_gap
        self.start_ground = start_ground
        self.end_ground = end_ground

    def draw(self, cell, origin, layer=0, datatype=0, draw_trace=False, draw_gap=True, draw_ground=False):
        """Draw the specified structure(s) into the given cell (if not None) and return a tuple of polygons.

        The boolean keywords `draw_trace`, `draw_gap`, and `draw_ground` can be used to draw any combination of
        the three possible structures. For example, the default values of `draw_trace=False`, `draw_gap=True`,
        `draw_ground=False` draws only the CPW gaps, which represent absence of metal. To draw instead the
        structures that represent metal in a CPW with specified ground planes, instantiate with `draw_ground` not
        equal to None and draw using `draw_trace=True`, `draw_gap=False`, `draw_ground=True`.

        :param cell: the cell into which to draw the structure, if not None.
        :type cell: gdspy.Cell or None
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :param bool draw_trace: if True, draw the center trace.
        :param bool draw_gap: if True, draw the gaps.
        :param bool draw_ground: if True, draw the ground planes; the instance must have been created with `ground`
                                 equal to a number, not the default of None.
        :return: the drawn structures, regardless of whether they were added to a cell.
        :rtype: tuple[gdspy.PolygonSet]
        """
        phi = np.arctan2(*self.span[::-1])  # The angle of the vector pointing from start to end
        rotation = np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
        # Draw the selected structures: create a tuple containing one or more lists of polygon vertices
        if draw_trace and draw_gap and draw_ground:
            vertices = ([(0, self.start_trace / 2 + self.start_gap + self.start_ground),
                         (self.length, self.end_trace / 2 + self.end_gap + self.end_ground),
                         (self.length, -(self.end_trace / 2 + self.end_gap + self.end_ground)),
                         (0, -(self.start_trace / 2 + self.start_gap + self.start_ground))],)
        elif draw_trace and draw_gap:
            vertices = ([(0, self.start_trace / 2 + self.start_gap),
                         (self.length, self.end_trace / 2 + self.end_gap),
                         (self.length, -(self.end_trace / 2 + self.end_gap)),
                         (0, -(self.start_trace / 2 + self.start_gap))],)
        elif draw_gap and draw_ground:
            upper = [(0, self.start_trace / 2 + self.start_gap + self.start_ground),
                     (self.length, self.end_trace / 2 + self.end_gap + self.end_ground),
                     (self.length, self.end_trace / 2 + self.end_gap),
                     (0, self.start_trace / 2)]
            lower = [(x, -y) for x, y in upper]
            vertices = (upper, lower)
        elif draw_trace and draw_ground:  # Positive CPW
            upper_ground = [(0, self.start_trace / 2 + self.start_gap + self.start_ground),
                            (self.length, self.end_trace / 2 + self.end_gap + self.end_ground),
                            (self.length, self.end_trace / 2 + self.end_gap),
                            (0, self.start_trace / 2 + self.start_gap)]
            lower_ground = [(x, -y) for x, y in upper_ground]
            trace = [(0, self.start_trace / 2),
                     (self.length, self.end_trace / 2),
                     (self.length, -self.end_trace / 2),
                     (0, -self.start_trace / 2)]
            vertices = (upper_ground, trace, lower_ground)
        elif draw_trace:
            vertices = ([(0, self.start_trace / 2),
                         (self.length, self.end_trace / 2),
                         (self.length, -self.end_trace / 2),
                         (0, -self.start_trace / 2)],)
        elif draw_gap:  # Negative CPW, the default
            upper = [(0, self.start_trace / 2 + self.start_gap),
                     (self.length, self.end_trace / 2 + self.end_gap),
                     (self.length, self.end_trace / 2),
                     (0, self.start_trace / 2)]
            lower = [(x, -y) for x, y in upper]
            vertices = (upper, lower)
        elif draw_ground:
            upper = [(0, self.start_trace / 2 + self.start_gap + self.start_ground),
                     (self.length, self.end_trace / 2 + self.end_gap + self.end_ground),
                     (self.length, self.end_trace / 2 + self.end_gap),
                     (0, self.start_trace / 2 + self.start_gap)]
            lower = [(x, -y) for x, y in upper]
            vertices = (upper, lower)
        else:  # Draw nothing
            vertices = tuple()
        # Create polygons using rotated and shifted vertices, and add them to the cell if given.
        polygons = list()
        for vertex_list in vertices:
            polygons.append(
                gdspy.Polygon(
                    points=[to_point(origin) + self.start + np.dot(rotation, to_point(vertex).T).T
                            for vertex in vertex_list],
                    layer=layer,
                    datatype=datatype)
            )
        if cell is not None:
            for polygon in polygons:
                cell.add(element=polygon)
        return tuple(polygons)
