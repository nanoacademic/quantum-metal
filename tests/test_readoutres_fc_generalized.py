import unittest
import numpy as np
from qiskit_metal import designs, Dict
from qiskit_metal.qlibrary.resonators.readoutres_fc_generalized import (
    GeneralizedReadoutResFC,
)


class TestGeneralizedReadoutResFC(unittest.TestCase):
    def setUp(self):
        self.design = designs.DesignFlipChip()
        self.design.overwrite_enabled = True

    def test_instantiation(self):
        """Test component instantiation and basic geometry check."""
        r1 = GeneralizedReadoutResFC(self.design, "R1")
        self.assertEqual(r1.name, "R1")
        # Check the qgeometry tables for `ro` and `ro_etch`.
        names = self.design.qgeometry.tables["poly"].name.values
        self.assertTrue(any(n.startswith("ro") for n in names))
        self.assertTrue(any(n.startswith("ro_etch") for n in names))
        self.assertIn("readout", r1.pins.keys())

    def test_options_fallback(self):
        """Test layer_subtract’s fallback to layer."""
        options = Dict(layer="3", layer_subtract=None)
        r1 = GeneralizedReadoutResFC(self.design, "R1", options=options)

        # Check layer for `ro`.
        poly_table = self.design.qgeometry.tables["poly"]
        ro_rows = poly_table[poly_table["name"].str.startswith("ro")]
        self.assertTrue(all(str(l) == "3" for l in ro_rows.layer.values))

        # Check layer for `ro_etch`.
        etch_rows = poly_table[poly_table["name"].str.startswith("ro_etch")]
        self.assertTrue(all(str(l) == "3" for l in etch_rows.layer.values))

    def test_mirroring(self):
        """Test mirroring."""
        r_orig = GeneralizedReadoutResFC(
            self.design, "R_orig", options=Dict(pos_x="0um", pos_y="0um")
        )
        # Middle of the CPW.
        orig_pin_mid = r_orig.pins["readout"]["middle"]

        # Mirror along x.
        r_mx = GeneralizedReadoutResFC(
            self.design, "R_mx", options=Dict(pos_x="0um", pos_y="0um", mirror_x=True)
        )
        mx_pin_mid = r_mx.pins["readout"]["middle"]
        # (x,y) → (-x,y).
        np.testing.assert_allclose(mx_pin_mid[0], -orig_pin_mid[0], atol=1e-9)
        np.testing.assert_allclose(mx_pin_mid[1], orig_pin_mid[1], atol=1e-9)

        # Mirror along y.
        r_my = GeneralizedReadoutResFC(
            self.design, "R_my", options=Dict(pos_x="0um", pos_y="0um", mirror_y=True)
        )
        my_pin_mid = r_my.pins["readout"]["middle"]
        # (x,y) → (x,-y).
        np.testing.assert_allclose(my_pin_mid[0], orig_pin_mid[0], atol=1e-9)
        np.testing.assert_allclose(my_pin_mid[1], -orig_pin_mid[1], atol=1e-9)

    def test_start_angle(self):
        """Test that different starting angles produce valid geometries."""
        for angle in [0, 45, 90, -45, -90]:
            with self.subTest(angle=angle):
                r = GeneralizedReadoutResFC(
                    self.design, f"R_{angle}", options=Dict(start_angle=angle)
                )
                self.assertIsNotNone(r)

    def test_invalid_start_angle(self):
        """Test that angles outside [-90, 90] raise `ValueError`."""
        for angle in [90.1, -90.1, 180, -180]:
            with self.subTest(angle=angle):
                with self.assertRaises(ValueError) as cm:
                    GeneralizedReadoutResFC(
                        self.design, f"R_err_{angle}", options=Dict(start_angle=angle)
                    )
                self.assertIn("Consider mirroring the geometry", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
