"""This module contains equations for calculating properties of co-planar waveguide transmission lines, such as their
inductance and capacitance per unit length, and classes for drawing them as GDSII structures.

The classes with name that start with 'Positive' draw the positive space, meaning that structures correspond to metal.
The classes with names that start with 'Negative' draw the negative space, meaning that structures correspond to the
absence of metal.
"""
from __future__ import absolute_import, division, print_function

import gdspy
import numpy as np
from scipy.constants import c, epsilon_0, mu_0, pi
from scipy.special import ellipk

from transmission_line.transmission_line import (DEFAULT_POINTS_PER_RADIAN, to_point, AbstractTransmissionLine, Segment,
                                                 SmoothedSegment)


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
# Classes that draw the negative space of CPW structures.

class NegativeCPW(SmoothedSegment):
    """The negative space of a segment of co-planar waveguide: structures are absence of metal."""

    def __init__(self, outline, trace, gap, radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """Instantiate without drawing in any cell.

        :param outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.gap = gap
        if radius is None:
            radius = trace / 2 + gap
        super(NegativeCPW, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
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
        trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, max_points=0, gdsii_path=False)
        gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, max_points=0, gdsii_path=False)
        polygon_set = gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=0, layer=layer, datatype=datatype)
        cell.add(element=polygon_set)
        return polygon_set


class NegativeCPWBlank(SmoothedSegment):
    """Negative co-planar waveguide with the center trace missing, that is, the negative space of the trace and gaps:
    structures are absence of metal.

    This class draws a single FlexPath that is the union of the trace and gaps. This is useful when the trace is on
    another layer or has a different datatype. The interface is the same as :class:`CPW` for compatibility.
    """

    def __init__(self, outline, trace, gap, radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """Instantiate without drawing in any cell.

        :param outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.gap = gap
        if radius is None:
            radius = trace / 2 + gap
        super(NegativeCPWBlank, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
                                               round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the drawn polygon set, which should contain one polygon.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the object that was drawn into the cell.
        :rtype: gdspy.PolygonSet
        """
        points = [to_point(origin) + point for point in self.points]
        path_set = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, layer=layer, datatype=datatype,
                                  max_points=0, gdsii_path=False).to_polygonset()
        cell.add(element=path_set)
        return path_set


class NegativeCPWElbowCoupler(SmoothedSegment):
    """Negative co-planar waveguide elbow coupler: structures are absence of metal."""

    def __init__(self, tip_point, elbow_point, joint_point, trace, gap, radius=None,
                 points_per_radian=DEFAULT_POINTS_PER_RADIAN,
                 round_to=None):
        """Instantiate without drawing in any cell.

        :param point tip_point: the open end of the coupler; the first point of the segment.
        :param point elbow_point: the point where the coupler turns away from the feedline; the middle point of the
                                  segment.
        :param point joint_point: the point where the coupler joins the rest of the transmission line; the last point of
                                  the segment.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.gap = gap
        if radius is None:
            radius = trace / 2 + gap
        super(NegativeCPWElbowCoupler, self).__init__(outline=[tip_point, elbow_point, joint_point], radius=radius,
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
        trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, max_points=0, gdsii_path=False)
        gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, max_points=0, gdsii_path=False)
        path_set = gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=0, layer=layer, datatype=datatype)
        cell.add(element=path_set)

        if round_tip:
            v = points[0] - points[1]
            theta = np.arctan2(v[1], v[0])
            round_set = gdspy.Round(center=points[0], radius=self.trace / 2 + self.gap, inner_radius=self.trace / 2,
                                    initial_angle=theta - np.pi / 2, final_angle=theta + np.pi / 2, max_points=0,
                                    layer=layer, datatype=datatype)
            cell.add(element=round_set)
        else:
            raise NotImplementedError("Need to code this up.")

        return path_set, round_set


class NegativeCPWElbowCouplerBlank(SmoothedSegment):
    """Negative co-planar waveguide elbow coupler with the center trace missing, that is, the negative space of the
    trace and gaps: structures are absence of metal.

    This class draws a single polygon that is the union of the trace and gaps. This is useful when the trace is on
    another layer or has a different datatype. The interface is the same as :class:`NegativeCPWElbowCoupler` for
    compatibility.
    """

    def __init__(self, tip_point, elbow_point, joint_point, trace, gap, radius=None,
                 points_per_radian=DEFAULT_POINTS_PER_RADIAN, round_to=None):
        """Instantiate without drawing in any cell.

        :param point tip_point: the open end of the coupler; the first point of the segment.
        :param point elbow_point: the point where the coupler turns away from the feedline; the middle point of the
                                  segment.
        :param point joint_point: the point where the coupler joins the rest of the transmission line; the last point of
                                  the segment.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.gap = gap
        if radius is None:
            radius = trace / 2 + gap
        super(NegativeCPWElbowCouplerBlank, self).__init__(outline=[tip_point, elbow_point, joint_point], radius=radius,
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
        path_set = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, layer=layer, datatype=datatype,
                                  max_points=0, gdsii_path=False).to_polygonset()
        cell.add(element=path_set)

        if round_tip:
            v = points[0] - points[1]
            theta = np.arctan2(v[1], v[0])
            round_set = gdspy.Round(center=points[0], radius=self.trace / 2 + self.gap, initial_angle=theta - np.pi / 2,
                                    final_angle=theta + np.pi / 2, max_points=0, layer=layer, datatype=datatype)
            cell.add(element=round_set)
        else:
            raise NotImplementedError("Need to code this up.")
        return path_set, round_set


class NegativeCPWTransition(Segment):
    """Negative transition between two sections of co-planar waveguide: structures are absence of metal.

    The points of this structure are (start_point, end_point).
    """

    def __init__(self, start_point, start_trace, start_gap, end_trace, end_point, end_gap, round_to=None):
        """Instantiate without drawing in any cell.

        The points of this structure are (start_point, end_point).

        :param point start_point: the start point of the transition, typically the end point of the previous section.
        :param start_trace: the trace width of the previous section.
        :param start_gap: the gap width of the previous section.
        :param end_point: the end point of the transition, typically the start point of the following section.
        :param end_trace: the trace width of the following section.
        :param end_gap: the gap width of the following section.
        :param round_to: see :class:`SmoothedSegment`.
        """
        super(NegativeCPWTransition, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_trace = start_trace
        self.start_gap = start_gap
        self.end_trace = end_trace
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
        upper = [(0, self.start_trace / 2),
                 (0, self.start_trace / 2 + self.start_gap),
                 (self.length, self.end_trace / 2 + self.end_gap),
                 (self.length, self.end_trace / 2)]
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


class NegativeCPWTransitionBlank(Segment):
    """Negative transition between two sections of co-planar waveguide: structures are absence of metal.

    The points of this structure are (start_point, end_point).

    This class draws a single polygon that is the union of the trace and gaps. This is useful when the trace is on
    another layer or has a different datatype. The interface is the same as :class:`NegativeCPWTransition` for
    compatibility.
    """

    def __init__(self, start_point, start_trace, start_gap, end_point, end_trace, end_gap, round_to=None):
        """Instantiate without drawing in any cell.

        The points of this structure are (start_point, end_point).

        :param point start_point: the start point of the transition, typically the end point of the previous section.
        :param start_trace: the trace width of the previous section.
        :param start_gap: the gap width of the previous section.
        :param end_point: the end point of the transition, typically the start point of the following section.
        :param end_trace: the trace width of the following section.
        :param end_gap: the gap width of the following section.
        :param round_to: see :class:`SmoothedSegment`.
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
        points = [(0, self.start_trace / 2 + self.start_gap),
                  (self.length, self.end_trace / 2 + self.end_gap),
                  (self.length, -self.end_trace / 2 - self.end_gap),
                  (0, -self.start_trace / 2 - self.start_gap)]
        points_rotated = [np.dot(rotation, to_point(p).T).T for p in points]
        points_rotated_shifted = [to_point(origin) + self.start + p for p in points_rotated]
        polygon = gdspy.Polygon(points=points_rotated_shifted, layer=layer, datatype=datatype)
        cell.add(element=polygon)
        return polygon

# Classes that draw the positive space of CPW structures with finite ground planes.

class PositiveCPW(SmoothedSegment):
    """The positive space of a segment of co-planar waveguide with finite ground planes: structures are metal."""

    def __init__(self, outline, trace, gap, ground, radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN,
                 round_to=None):
        """Instantiate without drawing in any cell.

        :param list[point] outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param float ground: the width of the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.gap = gap
        self.ground = ground
        if radius is None:
            radius = trace / 2 + gap
        super(PositiveCPW, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
                                          round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the drawn polygon set, which contains three polygons.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the object that was drawn into the cell.
        :rtype: gdspy.PolygonSet
        """
        points = [to_point(origin) + point for point in self.points]
        trace_flexpath = gdspy.FlexPath(points=points, width=self.trace, max_points=0, gdsii_path=False)
        gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, max_points=0,
                                      gdsii_path=False)
        ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground, max_points=0,
                                         gdsii_path=False)
        negative_polygon_set = gdspy.boolean(gap_flexpath, trace_flexpath, 'not', max_points=0, layer=layer,
                                             datatype=datatype)
        positive_polygon_set = gdspy.boolean(ground_flexpath, negative_polygon_set, 'not', max_points=0, layer=layer,
                                             datatype=datatype)
        cell.add(element=positive_polygon_set)
        return positive_polygon_set


class PositiveCPWBlank(SmoothedSegment):
    """The positive space of a segment of co-planar waveguide with finite ground planes, with the trace removed:
    structures are metal.

    This class draws only the ground planes, with no center trace. This is useful when the trace is on another layer or
    has a different datatype. The interface is the same as :class:`PositiveCPW` for compatibility.
    """

    def __init__(self, outline, trace, gap, ground, radius=None, points_per_radian=DEFAULT_POINTS_PER_RADIAN,
                 round_to=None):
        """Instantiate without drawing in any cell.

        :param list[point] outline: the vertices of the CPW path, before smoothing; see :func:`smooth`.
        :param float trace: the width of the center trace.
        :param float gap: the width of the gaps on each side of the center trace between it and the ground planes.
        :param float ground: the width of the ground planes.
        :param radius: see :func:`smooth`.
        :type radius: float or None
        :param int points_per_radian: see :func:`smooth`.
        :param float round_to: see :class:`SmoothedSegment`.
        """
        self.trace = trace
        self.gap = gap
        self.ground = ground
        if radius is None:
            radius = trace / 2 + gap
        super(PositiveCPWBlank, self).__init__(outline=outline, radius=radius, points_per_radian=points_per_radian,
                                               round_to=round_to)

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the drawn polygon set, which contains two polygons.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the object that was drawn into the cell.
        :rtype: gdspy.PolygonSet
        """
        points = [to_point(origin) + point for point in self.points]
        trace_and_gap_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap, max_points=0,
                                                gdsii_path=False)
        ground_flexpath = gdspy.FlexPath(points=points, width=self.trace + 2 * self.gap + 2 * self.ground, max_points=0,
                                         gdsii_path=False)
        polygon_set = gdspy.boolean(ground_flexpath, trace_and_gap_flexpath, 'not', max_points=0, layer=layer,
                                    datatype=datatype)
        cell.add(element=polygon_set)
        return polygon_set


class PositiveCPWTransition(Segment):
    """Positive transition between two sections of co-planar waveguide."""

    def __init__(self, start_point, start_trace, start_gap, start_ground, end_trace, end_point, end_gap, end_ground,
                 round_to=None):
        """Instantiate without drawing in any cell.

        The points of this structure are (start_point, end_point).

        :param point start_point: the start point of the transition, typically the end point of the previous section.
        :param start_trace: the trace width of the previous section.
        :param start_gap: the gap width of the previous section.
        :param start_ground: the width of both of the ground planes of the previous section.
        :param end_point: the end point of the transition, typically the start point of the following section.
        :param end_trace: the trace width of the following section.
        :param end_gap: the gap width of the following section.
        :param end_ground: the width of both of the ground planes of the following section.
        :param round_to: see :class:`SmoothedSegment`.
        """
        super(PositiveCPWTransition, self).__init__(points=[start_point, end_point], round_to=round_to)
        self.start_trace = start_trace
        self.start_gap = start_gap
        self.start_ground = start_ground
        self.end_trace = end_trace
        self.end_gap = end_gap
        self.end_ground = end_ground

    def draw(self, cell, origin, layer, datatype=0):
        """Draw this structure into the given cell and return the three drawn polygons, which are the center trace and
        two ground planes.

        :param gdspy.Cell cell: the cell into which to draw the structure.
        :param point origin: the points of the drawn structure are relative to this point.
        :param int layer: the layer on which to draw.
        :param int datatype: the GDSII datatype.
        :return: the three polygons that were drawn into the cell.
        :rtype: tuple[gdspy.Polygon]
        """
        v = self.end - self.start
        phi = np.arctan2(v[1], v[0])
        rotation = np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])
        trace = [(0, self.start_trace / 2),
                 (self.length, self.end_trace / 2),
                 (self.length, -self.end_trace / 2),
                 (0, -self.start_trace / 2)]
        upper = [(0, self.start_trace / 2 + self.start_gap),
                 (0, self.start_trace / 2 + self.start_gap + self.start_ground),
                 (self.length, self.end_trace / 2 + self.end_gap + self.end_ground),
                 (self.length, self.end_trace / 2 + self.end_gap)]
        lower = [(x, -y) for x, y in upper]
        trace_rotated = [np.dot(rotation, to_point(p).T).T for p in trace]
        upper_rotated = [np.dot(rotation, to_point(p).T).T for p in upper]
        lower_rotated = [np.dot(rotation, to_point(p).T).T for p in lower]
        trace_rotated_shifted = [to_point(origin) + self.start + p for p in trace_rotated]
        upper_rotated_shifted = [to_point(origin) + self.start + p for p in upper_rotated]
        lower_rotated_shifted = [to_point(origin) + self.start + p for p in lower_rotated]
        trace_polygon = gdspy.Polygon(points=trace_rotated_shifted, layer=layer, datatype=datatype)
        upper_polygon = gdspy.Polygon(points=upper_rotated_shifted, layer=layer, datatype=datatype)
        lower_polygon = gdspy.Polygon(points=lower_rotated_shifted, layer=layer, datatype=datatype)
        cell.add(element=trace_polygon)
        cell.add(element=upper_polygon)
        cell.add(element=lower_polygon)
        return trace_polygon,upper_polygon, lower_polygon
