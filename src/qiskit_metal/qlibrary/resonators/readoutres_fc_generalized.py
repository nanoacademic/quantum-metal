"""Generalization of ``ReadoutResFC``

This module defines the ``GeneralizedReadoutResFC`` component class, which is
an extension of the standard ``ReadoutResFC`` to support horizontal and/or
vertical mirroring (via ``mirror_x`` and ``mirror_y`` options) and a
customizable starting angle (via ``start_angle``).

"""

from qiskit_metal.qlibrary.resonators.readoutres_fc import ReadoutResFC
from qiskit_metal import draw
import shapely.affinity
import numpy as np


class GeneralizedReadoutResFC(ReadoutResFC):
    """.. image:: GeneralizedReadoutResFC.png

    Generalized flip-chip readout resonator.

    This component extends ``ReadoutResFC`` to allow for a custom starting
    angle for the takeoff leg (``start_angle``, counter-clockwise) and options
    to mirror the entire geometry (``mirror_x``, ``mirror_y``), which are
    applied lastly.

    Geometry overview:

    -  Circle centered at (``pos_x``, ``pos_y``) with radius
       ``readout_radius``.
    -  Straight line (length ``readout_l1``) at ``start_angle`` degrees
       (restricted to -90 to +90 degrees, counter-clockwise).
    -  ``(-90 - start_angle)``-degree arc.
    -  Vertical line (length ``readout_l2``).
    -  90-degree bend.
    -  Horizontal line (length ``readout_l3``).
    -  180-degree bend.
    -  Horizontal line (length ``readout_l4``).
    -  Five meandering horizontal lines (length ``readout_l5``) separated by
       ±180-degree arcs.

    The arc bend radius is ``readout_cpw_turnradius``, measured from the CPW
    center to the center of rotation. Lines and arcs form a CPW with width
    ``readout_cpw_width`` and gap ``readout_cpw_gap``.

    Since the start angle is restricted to -90 to +90 degrees, users may need
    to mirror the geometry to achieve specific configurations.
    Note that the mirroring operation is applied last.

    Tuning tips:

    -  Change coupling to the qubit by varying ``readout_radius``.
    -  Couple to the feedthrough line via the horizontal section of length
       ``readout_l3``.
    -  Adjust resonator frequency by varying ``readout_l5``.
    """

    # Inherit and extend standard options with our generalized parameters.
    default_options = ReadoutResFC.default_options.copy()
    default_options.update(
        # Fallback to layer if it is not explicitly overridden.
        layer_subtract=None,
        mirror_x=False,
        mirror_y=False,
        # Starting arm takeoff angle in degrees (counter-clockwise).
        start_angle=-45.0,
    )

    def make_ro(self):
        """Create the resonator’s coupling patch with a custom start angle."""
        # Access to parsed values from the user option.
        p = self.p

        # Access to chip name.
        chip = p.chip

        # Local variables.
        r = p.readout_radius
        w = p.readout_cpw_width
        g = p.readout_cpw_gap
        turnradius = p.readout_cpw_turnradius
        l_1 = p.readout_l1
        l_2 = p.readout_l2
        l_3 = p.readout_l3
        l_4 = p.readout_l4
        l_5 = p.readout_l5

        # Get custom start angle and convert it to radians.
        start_angle_deg = float(self.options.get("start_angle", -45.0))
        if not -90 <= start_angle_deg <= 90:
            raise ValueError(
                f"Invalid starting angle: {start_angle_deg}. "
                " It must be between -90 and 90 degrees."
                " Consider mirroring the geometry for other configurations."
            )
        theta = np.radians(-start_angle_deg)

        # Create the coupling patch in term of a circle.
        cppatch = draw.Point(0, 0).buffer(r)

        # Useful coordinates.
        # We usually have to round them as to avoid HXT 3D meshing breaking
        # due to OCC errors associated with boolean unions.
        # TODO: Investigate further this issue.
        x_1 = np.round(l_1 * np.cos(theta), 12)
        y_1 = np.round(-l_1 * np.sin(theta), 12)
        # Position `coord_center` at the correct location.
        coord_center_x = np.round(x_1 - turnradius * np.sin(theta), 12)
        coord_center_y = np.round(y_1 - turnradius * np.cos(theta), 12)
        coord_center = draw.Point(coord_center_x, coord_center_y)
        # The end of the arc should lie perfectly horizontal relative to the
        # centre.
        x_2 = np.round(coord_center_x + turnradius, 12)
        y_2 = np.round(coord_center_y, 12)

        coord_init = draw.Point(x_1, y_1)
        x_3, y_3 = x_2, y_2
        x_4 = np.round(x_3, 12)
        y_4 = np.round(y_3 - l_2, 12)
        x_5 = np.round(x_4 + turnradius, 12)
        y_5 = np.round(y_4, 12)
        coord_init1 = draw.Point(x_4, y_4)
        coord_center1 = draw.Point(x_5, y_5)
        x_6 = np.round(x_5, 12)
        y_6 = np.round(y_5 - turnradius, 12)
        x_7 = np.round(x_5 + l_3, 12)
        y_7 = np.round(y_6, 12)
        x_8 = np.round(x_7, 12)
        y_8 = np.round(y_7 + turnradius, 12)
        coord_init2 = draw.Point((x_7, y_7))
        coord_center2 = draw.Point((x_8, y_8))
        x_9 = np.round(x_8, 12)
        y_9 = np.round(y_8 + turnradius, 12)
        x_10 = np.round(x_8 - l_4, 12)
        y_10 = np.round(y_9, 12)
        x_11 = np.round(x_10, 12)
        y_11 = np.round(y_10 + turnradius, 12)
        coord_init3 = draw.Point((x_10, y_10))
        coord_center3 = draw.Point((x_11, y_11))
        arc3 = self.arc(coord_init3, coord_center3, -np.pi)
        x_12 = np.round(x_11, 12)
        y_12 = np.round(y_11 + turnradius, 12)
        x_13 = np.round(x_12 + l_5, 12)
        y_13 = np.round(y_12, 12)
        line12 = draw.LineString([(x_12, y_12), (x_13, y_13)])
        x_14 = np.round(x_13, 12)
        y_14 = np.round(y_13 + turnradius, 12)
        coord_init4 = draw.Point((x_13, y_13))
        coord_center4 = draw.Point((x_14, y_14))
        arc4 = self.arc(coord_init4, coord_center4, np.pi)

        # For the straight takeoff segment to connect smoothly to the circular
        # arc, the radial vector of the arc at `coord_init` must be
        # perpendicular to the line. That is, at angle (pi/2 - theta).
        arc_angle = -(np.pi / 2 - theta)

        # Handle the case where `arc_angle` is 0.
        # to avoid LineString errors with single points in Shapely.
        if abs(arc_angle) < 1e-12:
            # Skip the arc.
            first_arc = []
        else:
            first_arc = [self.arc(coord_init, coord_center, arc_angle)]

        # Line containing the dynamic-angle line and arc, as well as a short
        # straight segment for smooth subtraction.
        cparm_line = draw.shapely.ops.unary_union(
            [
                draw.LineString([(0, 0), coord_init]),
            ]
            + first_arc
            + [
                draw.LineString([(x_3, y_3), (x_4, y_4)]),
                self.arc(coord_init1, coord_center1, np.pi / 2),
                draw.LineString([(x_6, y_6), (x_7, y_7)]),
                self.arc(coord_init2, coord_center2, np.pi),
                draw.LineString([(x_9, y_9), (x_10, y_10)]),
                arc3,
                line12,
                arc4,
                draw.translate(line12, 0, 2 * turnradius),
                draw.translate(arc3, 0, 4 * turnradius),
                draw.translate(line12, 0, 4 * turnradius),
                draw.translate(arc4, 0, 4 * turnradius),
                draw.translate(line12, 0, 6 * turnradius),
                draw.translate(arc3, 0, 8 * turnradius),
                draw.translate(line12, 0, 8 * turnradius),
            ]
        )
        cparm = cparm_line.buffer(w / 2, cap_style=2, join_style=1)
        # Fix the gap resulting from buffer.
        eps = 1e-3
        cparm = draw.Polygon(cparm.exterior)
        cparm = cparm.buffer(eps, join_style=2).buffer(-eps, join_style=2)

        # Create combined objects for the signal line and the etch.
        ro = draw.shapely.ops.unary_union([cppatch, cparm])
        ro_etch = ro.buffer(g, cap_style=2, join_style=2)
        x_15, y_15 = x_14, y_14 + 7 * turnradius
        x_16, y_16 = x_15 + g / 2, y_15
        port_line = draw.LineString([(x_15, y_15 + w / 2), (x_15, y_15 - w / 2)])
        subtract_patch = draw.LineString(
            [(x_16, y_16 - w / 2 - g - eps), (x_16, y_16 + w / 2 + g + eps)]
        ).buffer(g / 2, cap_style=2)
        ro_etch = ro_etch.difference(subtract_patch)

        # Rotate and translate.
        polys = [ro, ro_etch, port_line]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)

        # Update each object.
        [ro, ro_etch, port_line] = polys

        # Generate `QGeometry`.
        layer_sub = p.layer_subtract
        if (
            layer_sub is None
            or str(layer_sub).strip() == ""
            or str(layer_sub).lower() == "none"
        ):
            layer_sub = p.layer

        self.add_qgeometry("poly", dict(ro=ro), chip=chip, layer=p.layer)
        self.add_qgeometry(
            "poly",
            dict(ro_etch=ro_etch),
            chip=chip,
            layer=layer_sub,
            subtract=p.subtract,
        )

        # Generate pins.
        self.add_pin("readout", port_line.coords, width=w, gap=g, chip=chip)

        # Apply mirroring if it is needed.
        # IDEA: This could be generalized to other components. It is possibly
        # quite useful for custom ones.
        mirror_x = self.options.get("mirror_x", False)
        mirror_y = self.options.get("mirror_y", False)
        if not mirror_x and not mirror_y:
            return

        comp_id = self.id
        cx, cy = self.design.parse_value([self.options.pos_x, self.options.pos_y])

        # Scaling factors.
        xfact = -1.0 if mirror_x else 1.0
        yfact = -1.0 if mirror_y else 1.0

        # Mirror all generated polygons in the design tables (`ro` and
        # `ro_etch` here).
        for table_name, table in self.design.qgeometry.tables.items():
            mask = table["component"] == comp_id
            table.loc[mask, "geometry"] = table.loc[mask, "geometry"].apply(
                lambda g: shapely.affinity.scale(
                    g, xfact=xfact, yfact=yfact, origin=(cx, cy)
                )
            )

        # Mirror the connection pins such that the connected CPW routes will
        # snap to the correct side.
        for pin_name, pin in self.pins.items():
            # Mirror the 2D points of the pin line string.
            p_line = draw.LineString(pin["points"])
            p_line_mirrored = shapely.affinity.scale(
                p_line, xfact=xfact, yfact=yfact, origin=(cx, cy)
            )
            pin["points"] = np.array(p_line_mirrored.coords)

            # Mirror the middle point of the pin.
            p_mid = draw.Point(pin["middle"])
            p_mid_mirrored = shapely.affinity.scale(
                p_mid, xfact=xfact, yfact=yfact, origin=(cx, cy)
            )
            pin["middle"] = np.array(p_mid_mirrored.coords[0])

            # Scale the directional vectors.
            pin["normal"][0] *= xfact
            pin["normal"][1] *= yfact
            pin["tangent"][0] *= xfact
            pin["tangent"][1] *= yfact
