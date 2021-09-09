"""Thin-film superconducting circuits are sensitive to the presence of magnetic flux vortices, which can become
trapped (even in type-I films) as the film is cooled below the superconducting transition temperature. To reduce the
unwanted effects of such vortices, such as dissipation at microwave frequencies, a standard practice is to add a
'mesh' of holes in the superconducting ground plane through which the magnetic flux lines can pass instead of forming
vortices.

This module contains functions that calculate the centers of a hole mesh around various types of Segments defined
elsewhere in the library. Because the ideal mesh dimensions (of order ten microns) are much smaller than typical
superconducting microwave resonators (of order ten millimeters), a well-designed mesh may include thousands of holes per
resonator. Adding such a mesh to an entire device would produce a large GDS file, and the functions here can be used to
create the dense mesh only next to the sensitive devices, where it is needed.
"""
import numpy as np

import transmission_line as tl


def smoothed_segment(segment, center_to_first_row, mesh_spacing, num_mesh_rows, at_least_one_column=False):
    """Calculate and return the center points of a mesh surrounding a SmoothedSegment.

    :param tl.SmoothedSegment segment:
    :param float center_to_first_row: the distance, in the units used by the segment, from the center of the segment to
                                      the center of the first mesh row.
    :param float mesh_spacing: the approximate spacing between all mesh points, in the units used by the segment.
    :param int num_mesh_rows: the number of rows of mesh in the direction perpendicular to the segment.
    :param bool at_least_one_column: if True, segments shorter than the mesh spacing will still have one corresponding
                                     mesh column; otherwise, they will not.
    :return: a list of mesh center points.
    :rtype: list[point]
    """
    mesh_centers = list()
    # Mesh the straight sections
    starts = [segment.start] + [bend[-1] for bend in segment.bends]
    ends = [bend[0] for bend in segment.bends] + [segment.end]
    for start, end in zip(starts, ends):
        v = end - start
        length = np.linalg.norm(v)
        phi = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
        num_mesh_columns = max(int(np.floor(length / mesh_spacing)), int(at_least_one_column))
        if num_mesh_columns == 0:
            continue
        elif num_mesh_columns == 1:
            x = np.array([length / 2])
        else:
            x = np.linspace(mesh_spacing / 2, length - mesh_spacing / 2, num_mesh_columns)
        y = center_to_first_row + mesh_spacing * np.arange(num_mesh_rows)
        xx, yy = np.meshgrid(np.concatenate((x, x)), np.concatenate((y, -y)))
        Rxy = np.dot(R, np.vstack((xx.flatten(), yy.flatten())))
        mesh_centers.extend(zip(start[0] + Rxy[0, :], start[1] + Rxy[1, :]))
    # Mesh the curved sections
    for row in range(num_mesh_rows):
        center_to_row = center_to_first_row + row * mesh_spacing
        for radius in [segment.radius - center_to_row, segment.radius + center_to_row]:
            if radius < mesh_spacing / 2:
                continue
            for angle, corner, offset in zip(segment.angles, segment.corners, segment.offsets):
                num_points = int(np.round(radius * np.abs(angle) / mesh_spacing))
                if num_points == 1:
                    max_angle = 0
                else:
                    max_angle = (1 - 1 / num_points) * angle / 2
                mesh_centers.extend([corner + offset +
                                     radius * np.array([np.cos(phi), np.sin(phi)]) for phi in
                                     (np.arctan2(-offset[1], -offset[0]) +
                                      np.linspace(-max_angle, max_angle, num_points))])
    return mesh_centers


def transition(segment, start_center_to_first_row, end_center_to_first_row, mesh_spacing, num_mesh_rows,
               at_least_one_column=False):
    """Calculate and return the center points of a mesh surrounding a single trapezoidal Segment that forms a transition
    between segments with different widths.

    :param tl.Segment segment: the Segment to surround with mesh.
    :param float start_center_to_first_row: the distance, in the units used by the segment, from the center of the
                                            segment to the center of the first mesh row at the starting side.
    :param float end_center_to_first_row: the distance, in the units used by the segment, from the center of the segment
                                          to the center of the first mesh row at the ending side.
    :param float mesh_spacing: the approximate spacing between all mesh points, in the units used by the segment.
    :param int num_mesh_rows: the number of rows of mesh in the direction perpendicular to the segment.
    :param bool at_least_one_column: if True, transitions shorter than the mesh spacing will still have one
                                     corresponding mesh column; otherwise, they will not.
    :return: a list of mesh center points.
    :rtype: list[point]
    """
    mesh_centers = list()
    v = segment.end - segment.start
    length = np.linalg.norm(v)
    phi = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    num_mesh_columns = max(int(np.floor(length / mesh_spacing)), int(at_least_one_column))
    if num_mesh_columns == 0:
        return mesh_centers
    elif num_mesh_columns == 1:
        x = np.array([length / 2])
    else:
        x = np.linspace(mesh_spacing / 2, length - mesh_spacing / 2, num_mesh_columns)
    y = mesh_spacing * np.arange(num_mesh_rows, dtype=np.float)  # ToDo: why float here?
    xxp, yyp = np.meshgrid(x, y)  # These correspond to the positive y-values
    y_shift = start_center_to_first_row + (end_center_to_first_row - start_center_to_first_row) * x / length
    yyp += y_shift
    xx = np.concatenate((xxp, xxp))
    yy = np.concatenate((yyp, -yyp))  # The negative y-values are reflected
    Rxy = np.dot(R, np.vstack((xx.flatten(), yy.flatten())))
    mesh_centers.extend(zip(segment.start[0] + Rxy[0, :], segment.start[1] + Rxy[1, :]))
    return mesh_centers


def rounded_open(segment, center_to_first_row, mesh_spacing, num_mesh_rows, at_least_one_column=False, extend=0):
    """Calculate and return the center points of a mesh surrounding a single RoundedOpen Segment that forms the end of a
    transmission line.

    :param tl.Segment segment: the Segment to surround with mesh.
    :param float center_to_first_row: the distance, in the units used by the segment, from the center of the segment to
                                      the center of the first mesh row.
    :param float mesh_spacing: the approximate spacing between all mesh points, in the units used by the segment.
    :param int num_mesh_rows: the number of rows of mesh in the direction perpendicular to the segment.
    :param bool at_least_one_column: if True, transitions shorter than the mesh spacing will still have one
                                     corresponding mesh column; otherwise, they will not.
    :param float extend: continue the mesh this distance past the open end.
    :return: a list of mesh center points.
    :rtype: list[point]
    """
    # ToDo: this code is copied from transition above; maybe it can be shared or simplified
    mesh_centers = list()
    if segment.open_at_start:
        ultimate, penultimate = segment.points
    else:
        penultimate, ultimate = segment.points
    v = ultimate - penultimate  # This vector points toward the open end
    length = np.linalg.norm(v) + extend
    phi = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    num_mesh_columns = max(int(np.floor(length / mesh_spacing)), int(at_least_one_column))
    if num_mesh_columns == 0:
        return mesh_centers
    elif num_mesh_columns == 1:
        x = np.array([length / 2])
    else:
        x = np.linspace(mesh_spacing / 2, length - mesh_spacing / 2, num_mesh_columns)
    y = center_to_first_row + mesh_spacing * np.arange(num_mesh_rows, dtype=np.float)
    xxp, yyp = np.meshgrid(x, y)  # These correspond to the positive y-values
    xx = np.concatenate((xxp, xxp))
    yy = np.concatenate((yyp, -yyp))  # The negative y-values are reflected
    Rxy = np.dot(R, np.vstack((xx.flatten(), yy.flatten())))
    mesh_centers.extend(zip(segment.start[0] + Rxy[0, :], segment.start[1] + Rxy[1, :]))
    return mesh_centers
