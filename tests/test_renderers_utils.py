import unittest
import numpy as np
import shapely
from qiskit_metal import designs, Dict
from qiskit_metal.qlibrary.sample_shapes.rectangle import Rectangle
from qiskit_metal.renderers.utils import get_combined_dimensions


class TestRenderersUtils(unittest.TestCase):
    def setUp(self):
        self.design = designs.DesignPlanar()
        self.design.overwrite_enabled = True

    def test_get_combined_dimensions(self):
        """Simple check of the resulting combined dimensions for a trivial design."""
        r1 = Rectangle(
            self.design,
            "R1",
            options=Dict(pos_x="0.5mm", pos_y="0.5mm", width="1mm", height="1mm"),
        )
        r2 = Rectangle(
            self.design,
            "R2",
            options=Dict(pos_x="2.5mm", pos_y="1.5mm", width="1mm", height="1mm"),
        )

        expected_bounds = (0.0, 0.0, 3.0, 2.0)

        dims = get_combined_dimensions(r1, r2)
        np.testing.assert_allclose(dims["bounds"], expected_bounds, atol=1e-12)

        self.assertAlmostEqual(dims["width"], expected_bounds[2] - expected_bounds[0])
        self.assertAlmostEqual(dims["height"], expected_bounds[3] - expected_bounds[1])

        geoms = []
        comp_ids = {r1.id, r2.id}
        for table in self.design.qgeometry.tables.values():
            mask = table["component"].isin(comp_ids)
            geoms.extend(table.loc[mask, "geometry"].tolist())
        truth_bounds = shapely.unary_union(geoms).bounds
        np.testing.assert_allclose(dims["bounds"], truth_bounds, atol=1e-12)

        self.assertAlmostEqual(dims["width"], truth_bounds[2] - truth_bounds[0])
        self.assertAlmostEqual(dims["height"], truth_bounds[3] - truth_bounds[1])

    def test_get_combined_dimensions_empty(self):
        """Test `get_combined_dimensions` with no components."""
        dims = get_combined_dimensions()
        self.assertEqual(
            dims, {"width": 0.0, "height": 0.0, "bounds": (0.0, 0.0, 0.0, 0.0)}
        )

    def test_get_combined_dimensions_invalid_components(self):
        """Test `get_combined_dimensions` with invalid components."""

        class FakeComponent:
            pass

        component_bad = FakeComponent()
        dims = get_combined_dimensions(component_bad)
        self.assertEqual(
            dims, {"width": 0.0, "height": 0.0, "bounds": (0.0, 0.0, 0.0, 0.0)}
        )

        component_good = Rectangle(self.design, "R1")
        dims_mixed = get_combined_dimensions(component_bad, component_good)
        # Should give us a non-trivial result.
        self.assertGreater(dims_mixed["width"], 0)
        self.assertGreater(dims_mixed["height"], 0)

    def test_get_combined_dimensions_formatting(self):
        """Test formatting and unit conversion in `get_combined_dimensions`."""
        r1 = Rectangle(
            self.design,
            "R1",
            options=Dict(pos_x="0um", pos_y="0um", width="1mm", height="1mm"),
        )

        dims_mm = get_combined_dimensions(r1, unit="mm")
        self.assertTrue(str(dims_mm["width"]).endswith("mm"))

        dims_um = get_combined_dimensions(r1, unit="um")
        self.assertTrue(str(dims_um["width"]).endswith("um"))

        dims_nm = get_combined_dimensions(r1, unit="nm")
        self.assertTrue(str(dims_nm["width"]).endswith("nm"))

        width_mm = get_combined_dimensions(r1)["width"]
        width_nm_str = get_combined_dimensions(r1, unit="nm")["width"]
        width_nm = float(width_nm_str.split()[0])
        np.testing.assert_allclose(width_nm, width_mm * 1e6, rtol=1e-8)

    def test_invalid_unit(self):
        """Test that an invalid unit raises a `ValueError`."""
        r1 = Rectangle(self.design, "R1")
        invalid_unit = "cm"
        with self.assertRaises(ValueError) as cm:
            get_combined_dimensions(r1, unit=invalid_unit)
        self.assertEqual(
            str(cm.exception),
            f"Invalid unit: {invalid_unit}. Supported units are 'nm', 'um', 'mm', 'm'.",
        )


if __name__ == "__main__":
    unittest.main()
