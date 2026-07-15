import unittest
import json
import os
from qiskit_metal import designs, Dict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.renderers.renderer_qtcad.qtcad_renderer import QQTCADRenderer


class TestQQTCADRendererFlipChip(unittest.TestCase):
    def setUp(self):
        self.design = designs.DesignFlipChip()
        self.design.overwrite_enabled = True

    def test_auto_layer_classification(self):
        """Verify that the active layers are automatically categorized as metal."""
        TransmonCross(self.design, "Q1", options=Dict(chip="Q_chip", layer=1))
        TransmonCross(self.design, "Q2", options=Dict(chip="C_chip", layer=2))

        qtcad_renderer = QQTCADRenderer(self.design)

        self.assertIn(1, qtcad_renderer.layer_types["metal"])
        self.assertIn(2, qtcad_renderer.layer_types["metal"])
        # For completeness, check the layer with the dielectrics.
        self.assertIn(3, qtcad_renderer.layer_types["dielectric"])


if __name__ == "__main__":
    unittest.main()
