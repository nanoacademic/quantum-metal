import unittest
import numpy as np
from qiskit_metal import designs, Dict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.renderers.renderer_gmsh.gmsh_renderer import QGmshRenderer


class TestQGmshRendererFlipChip(unittest.TestCase):
    def setUp(self):
        self.design = designs.DesignFlipChip()
        self.design.overwrite_enabled = True
        self.renderer = QGmshRenderer(self.design)

        # Configure chips with different elevations.
        self.design.chips["Q_chip"]["size"]["center_z"] = "11um"
        self.design.chips["C_chip"]["size"]["center_z"] = "0um"

        # Add xmons to both chips on separate layers.
        TransmonCross(self.design, "Q-xmon", options=Dict(chip="Q_chip", layer="1"))
        TransmonCross(
            self.design, "C-xmon1", options=Dict(chip="C_chip", layer="2", pos_x="1mm")
        )
        TransmonCross(
            self.design, "C-xmon2", options=Dict(chip="C_chip", layer="2", pos_x="-1mm")
        )

    def test_substrate_rendering_presence(self):
        """Verify that dielectric holders are created."""
        self.renderer.render_design(mesh_geoms=False)

        # Check if `chip_substrates` was populated.
        self.assertTrue(len(self.renderer.chip_substrates) > 0)
        self.assertIn("Q_chip", self.renderer.chip_substrates)
        self.assertIn("C_chip", self.renderer.chip_substrates)

        # Check for the dielectric physical groups.
        substrate_layer = -2
        self.assertIn(substrate_layer, self.renderer.physical_groups)
        substrate_groups = self.renderer.physical_groups[substrate_layer]
        self.assertIn("dielectric_Q_chip", substrate_groups)
        self.assertIn("dielectric_C_chip", substrate_groups)

    def test_z_separation_components_and_planes(self):
        """Verify that components and planes have different z-coordinates."""
        self.renderer.render_design(mesh_geoms=False)

        # Q-chip.
        thick1, z1 = self.renderer.get_thickness_zcoord_for_layer_datatype(1)
        # 11 μm = 0.01 mm
        self.assertAlmostEqual(z1, 0.011)

        # C-chip.
        thick2, z2 = self.renderer.get_thickness_zcoord_for_layer_datatype(2)
        self.assertAlmostEqual(z2, 0.0)

        self.assertNotEqual(z1, z2)

    def test_components_and_ground_planes(self):
        """Verify if components and ground planes are rendered."""

        self.renderer.render_design(mesh_geoms=False)

        # self.renderer.physical_groups:
        # {
        #     1: {"Q-xmon_cross": 1, "Q-xmon_rect_jj": 2, "ground_plane_(layer 1)": 3},
        #     2: {
        #         "C-xmon1_cross": 4,
        #         "C-xmon2_cross": 5,
        #         "C-xmon1_rect_jj": 6,
        #         "C-xmon2_rect_jj": 7,
        #         "ground_plane_(layer 2)": 8,
        #     },
        #     -2: {
        #         "dielectric_C_chip": 9,
        #         "dielectric_C_chip_sfs": 10,
        #         "dielectric_Q_chip": 11,
        #         "dielectric_Q_chip_sfs": 12,
        #     },
        #     "global": {"vacuum_box": 13, "vacuum_box_sfs": 14},
        # }

        # Layers.
        self.assertIn(1, self.renderer.physical_groups)
        self.assertIn(2, self.renderer.physical_groups)

        # Ground planes.
        self.assertIn("ground_plane_(layer 1)", self.renderer.physical_groups[1])
        self.assertIn("ground_plane_(layer 2)", self.renderer.physical_groups[2])

        # Components.
        self.assertIn("Q-xmon_cross", self.renderer.physical_groups[1])
        self.assertIn("C-xmon1_cross", self.renderer.physical_groups[2])
        self.assertIn("C-xmon2_cross", self.renderer.physical_groups[2])


if __name__ == "__main__":
    unittest.main()
