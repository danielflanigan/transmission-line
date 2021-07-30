"""This example demonstrates drawing both positive and negative CPW using the cpw.py module."""
import os

import numpy as np
import gdspy
from transmission_line import transmission_line as tl, cpw


def main(output_directory):
    """Draw a design demonstrating most features of `transmission_line/cpw.py` and save it in the given directory as
    `cpw.gds`.
    """
    library = gdspy.GdsLibrary(name='cpw')
    main_cell = library.new_cell('main')

    # Draw a single section of CPW. Use boolean keywords to draw the negative space, so structures represent absence
    # of metal. By default, bends that preserve the gaps replace the sharp corners. This means that the CPW starts
    # and ends at the first and last outline points, but it does not actually pass through the intermediate points.
    single_cpw = cpw.CPW(outline=[(100, 0), (100, -100), (300, -300), (400, -100)],
                         trace=10,  # The width of the center trace
                         gap=20,  # The width of the gaps
                         ground=None,  # The default
                         radius=50,  # The bends have a reasonable default radius, which also can be set here
                         )
    print(single_cpw.draw(cell=main_cell,  # Draw the CPW into this cell
                    origin=(0, -100),  # The points are drawn relative to this point, so this CPW starts at (100, -100)
                    layer=0,
                    draw_trace=False,  # The default
                    draw_gap=True,  # The default
                    draw_ground=False))  # The default; if True, the CPW must be instantiated with ground=<a number>.
    # Draw the same CPW at a different place on a different layer.
    print(single_cpw.draw(cell=main_cell,
                    origin=(700, -100),
                    layer=1))

    # Draw a feedline with launches for wirebond connections to an external circuit.

    # Use these variables to parameterize the feedline and launches.
    feedline_trace = 20  # The width of the center trace
    feedline_gap = 10  # The width of each of the gaps
    feedline_ground = 20  # The width of each of the ground planes
    launch_trace = 400
    launch_gap = 200
    launch_ground = 400
    launch_border = 50
    launch_length = 450
    launch_transition = 500

    # Draw a feedline, meaning a CPW that expands at each end to form launches that can be connected to CPW off the
    # chip. Draw only the center trace and the ground planes, namely the structures that represent metal. Since the
    # grounds are drawn here, their width must be given. Note that the all of the structures have (0, 0) as their
    # first point. When they are assembled in order and drawn, they will be connected head to tail because the code
    # will draw them so that the end point of one structure is the beginning of the next structure.

    # To set the center trace back from the edge of the chip, a good practice, start with a section of CPW that will
    # be drawn as a blank, with all `draw_something` keywords set to False.
    left_border = cpw.CPW(outline=[(0, 0), (launch_border, 0)],  # The direction is rightward
                          trace=launch_trace, gap=launch_gap, ground=launch_ground)  # These values are not used
    left_launch = cpw.CPW(outline=[(0, 0), np.array([launch_length, 0])],  # Still going right
                          trace=launch_trace, gap=launch_gap, ground=launch_ground)
    left_transition = cpw.CPWTransition(start_point=(0, 0), end_point=np.array([launch_transition, 0]),
                                        start_trace=launch_trace, end_trace=feedline_trace,
                                        start_gap=launch_gap, end_gap=feedline_gap,
                                        start_ground=launch_ground, end_ground=feedline_ground)
    feedline_cpw = cpw.CPW(
        outline=[(0, 0), (2000, 0), (2000, 1000)],  # This segment bends upward
        trace=feedline_trace, gap=feedline_gap, ground=feedline_ground,
        radius=100)  # Set the radius of the bends in the CPW center trace; the gap is preserved in the bends.
    upper_transition = cpw.CPWTransition(start_point=(0, 0), end_point=np.array([0, launch_transition]),  # upward
                                        start_trace=feedline_trace, end_trace=launch_trace,
                                        start_gap=feedline_gap, end_gap=launch_gap,
                                        start_ground=feedline_ground, end_ground=launch_ground)
    upper_launch = cpw.CPW(outline=[(0, 0), np.array([0, launch_length])],  # Also going upward
                           trace=launch_trace, gap=launch_gap, ground=launch_ground)
    upper_border = cpw.CPW(outline=[(0, 0), (0, launch_border)],  # Also going upward
                           trace=launch_trace, gap=launch_gap, ground=launch_ground)
    # Assemble the Segments in the proper order so that they can be drawn head-to-tail.
    feedline = tl.SegmentList([left_border, left_launch, left_transition, feedline_cpw, upper_transition,
                               upper_launch, upper_border])

    print(f"The start point of the feedline is {feedline.start}")
    print(f"The end point of the feedline is {feedline.end}")
    print(f"The span of the feedline is {feedline.span}")
    print(f"The bounding box of the feedline is defined by {feedline.bounds[0]} at lower left "
          f"and {feedline.bounds[1]} at upper right.")
    print(f"The total feedline length, including bends, is {feedline.length}")
    # Draw the feedline in the main cell on the given layer, with the start point at the given origin. This puts the
    # bend in this CPW near (0, 0). Draw the trace and ground, but not the gap, of all Segments except for the first and
    # last, for which no structures are drawn because the individual keywords override the global keywords.
    feedline.draw(cell=main_cell, origin=(-3000, 0), layer=0,
                  draw_trace=True,
                  draw_gap=False,
                  draw_ground=True
                  # individual_keywords={0: dict(draw_trace=False, draw_ground=False),  # left_border
                  #                     -1: dict(draw_trace=False, draw_ground=False)},  # upper_border
                  )

    """
    # Draw another negative CPW with an elbow coupler next to the feedline. This structure is a section of CPW that has
    # exactly three points and by default draws a rounded cap that avoids sharp edges. It currently has to be the first
    # element of the SegmentList, but this will be changed soon.
    coupled_cpw_trace = 10
    coupled_cpw_gap = 20
    coupled_cpw = tl.SegmentList([
        cpw.NegativeCPWElbowCoupler(tip_point=(0, 100),  # The tip of the coupler
                                    elbow_point=(0, 0),  # The coupler elbow, where it bends away from the feedline
                                    joint_point=(100, 0),  # The point where the coupler joins the next segment
                                    trace=coupled_cpw_trace,
                                    gap=coupled_cpw_gap),
        cpw.NegativeCPW(outline=[(0, 0), (500, 0), (400, 500)], trace=coupled_cpw_trace, gap=coupled_cpw_gap)
    ])
    # Draw the CPW with its elbow coupler next to the feedline. Using variables to calculate distances makes it easy to
    # update the design quickly. Because the elbow point given above is (0, 0), that point will be placed at the given
    # origin, next to the feedline.
    coupled_cpw_distance = feedline_trace / 2 + feedline_gap + feedline_ground + coupled_cpw_gap + coupled_cpw_trace / 2
    coupled_cpw.draw(cell=main_cell, origin=(coupled_cpw_distance, 300), layer=2)

    # Draw another negative CPW with an elbow coupler next to the feedline and an open at the opposite end. This could
    # be used as a half-wave resonator.
    half_wave_cpw = tl.SegmentList([
        cpw.NegativeCPWElbowCoupler(tip_point=(100, 0),  # The tip of the coupler
                                    elbow_point=(0, 0),  # The coupler elbow, where it bends away from the feedline
                                    joint_point=(0, 100),  # The point where the coupler joins the next segment
                                    trace=coupled_cpw_trace,
                                    gap=coupled_cpw_gap),
        cpw.NegativeCPW(outline=[(0, 0), (0, 500), (300, 500), (300, 700)], trace=coupled_cpw_trace,
                        gap=coupled_cpw_gap),
        # By default, this is open at the end point.
        cpw.NegativeCPWRoundedOpen(start_point=(0, 0), end_point=(0, 100), trace=coupled_cpw_trace, gap=coupled_cpw_gap)
    ])
    half_wave_cpw.draw(cell=main_cell, origin=(-1500, coupled_cpw_distance), layer=2)
    """

    # Save the gds file and return the library object
    gds_filename = 'cpw.gds'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    full_filename = os.path.join(output_directory, gds_filename)
    library.write_gds(outfile=full_filename)  # This will overwrite an existing file!
    print("gdspy saved {}".format(full_filename))
    return library


if __name__ == '__main__':
    library = main(output_directory='.')
    gdspy.LayoutViewer(library)
    print("The gdspy.GdsLibrary object is called 'library'")
