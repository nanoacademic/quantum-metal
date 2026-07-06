"""Unit tests for QQTCADRenderer: standalone QTCAD script generation."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from qiskit_metal.renderers.renderer_qtcad.qtcad_renderer import QQTCADRenderer


class TestQQTCADRendererExportScript(unittest.TestCase):
    """Unit tests for the `export_script` method in QQTCADRenderer."""

    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.test_dir.name)

        # Create dummy simulation parameters.
        self.dummy_data = {
            "gmsh_physical_groups": {
                "layer1": {
                    "dielectric_substrate": 1,
                    "ground_plane_sfs": 2,
                    "conductor1_sfs": 3
                }
            },
            "_conductors": {
                "gnd": ["ground_plane"],
                "q1": ["conductor1"]
            },
            "_inductive_ports": {
                "qubit1": {
                    "bnd_spec": "conductor1_sfs",
                    "inductance": 10e-9,
                    "length": 1e-6,
                    "width": 2e-6,
                    "dir": [1, 0, 0]
                }
            },
            "sample_holder": True,
            "qtcad_options": {
                "mesh_filepath": "dummy_mesh.msh4",
                "mesh_scale": 1e-3,
                "geo_filepath": "dummy_device.xao",
                "output_dir": str(self.test_dir_path),
                "make_subdir": False,
                "capacitance": {
                    "tol_rel": 0.05,
                    "tol_abs": 1e-15,
                    "min_converged_iters": 3
                },
                "capacitance_raw": None,
                "maxwell_emode": {
                    "num_modes": 3,
                    "tol_rel": 0.01,
                    "min_converged_iters": 5
                },
                "maxwell_emode_raw": None
            }
        }

        # Write dummy data to a JSON file.
        self.json_filepath = self.test_dir_path / "dummy_qtcad_data.json"
        with open(self.json_filepath, "w") as f:
            json.dump(self.dummy_data, f, indent=2)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_export_script_cap(self):
        """Test calling `export_script` for capacitance-matrix extraction."""
        script_filepath = self.test_dir_path / "run_cap_test.py"

        result_path = QQTCADRenderer.export_script(
            solve_for="cap",
            json_filepath=self.json_filepath,
            script_filepath=script_filepath
        )

        self.assertEqual(result_path, script_filepath)
        self.assertTrue(script_filepath.exists())

        # Verify script’s content.
        with open(script_filepath, "r") as f:
            content = f.read()

        # Check key elements related to the capacitance-matrix solver.
        self.assertIn("device = qtcad_device(mesh)", content)
        self.assertIn("device.new_dirichlet_bnd(boundary, 0)", content)
        self.assertIn("SolverCap", content)
        self.assertIn("solver_params_cap.tol_rel = 0.05", content)

        # Ensure no elements related to the Maxwell-eigenmode solver are present.
        self.assertNotIn("SolverEig", content)
        self.assertNotIn("new_pec_bnd", content)
        self.assertNotIn("solver_params_eig", content)

    def test_export_script_eigs(self):
        """Test calling `export_script` for Maxwell-eigenmodes simulations."""
        script_filepath = self.test_dir_path / "run_eigs_test.py"

        result_path = QQTCADRenderer.export_script(
            solve_for="eigs",
            json_filepath=self.json_filepath,
            script_filepath=script_filepath
        )

        self.assertEqual(result_path, script_filepath)
        self.assertTrue(script_filepath.exists())

        # Verify script’s content.
        with open(script_filepath, "r") as f:
            content = f.read()

        # Check key elements related to the Maxwell-eigenmode solver.
        self.assertIn("device = qtcad_device(mesh)", content)
        self.assertIn("device.new_pec_bnd(boundary)", content)
        self.assertIn("SolverEig", content)
        self.assertIn("solver_params_eig.num_modes = 3", content)

        # Ensure no elements related to the capacitance-matrix solver are present.
        self.assertNotIn("SolverCap", content)
        self.assertNotIn("new_dirichlet_bnd", content)
        self.assertNotIn("solver_params_cap", content)

    def test_export_script_invalid_solve_for(self):
        """Test that an invalid ``solve_for`` value raises ``ValueError``."""
        with self.assertRaises(ValueError) as context:
            QQTCADRenderer.export_script(
                solve_for="invalid",
                json_filepath=self.json_filepath
            )
        self.assertIn(
            "must be either 'cap' or 'eigs'",
            str(context.exception)
        )

    def test_export_script_missing_json_file(self):
        """Test that a missing JSON file raises ``FileNotFoundError``."""
        missing_path = self.test_dir_path / "non_existent.json"
        with self.assertRaises(FileNotFoundError) as context:
            QQTCADRenderer.export_script(
                solve_for="cap",
                json_filepath=missing_path
            )
        self.assertIn(
            "Please make sure to run `export_parameters()` first",
            str(context.exception)
        )

    def test_export_script_static_default_script_filepath(self):
        """Test `export_script` default Python path when none is provided."""
        result_path = QQTCADRenderer.export_script(
            "cap",
            self.json_filepath
        )

        expected_path = self.json_filepath.parent / "dummy_mesh_qtcad_only.py"
        self.assertEqual(result_path, expected_path)
        self.assertTrue(expected_path.exists())

    def test_export_script_from_renderer_instance(self):
        """Test calling `export_script` with a QQTCADRenderer instance."""
        # Instantiate QQTCADRenderer using __new__ to avoid side effects
        renderer = object.__new__(QQTCADRenderer)
        renderer.json_filepath = self.json_filepath

        script_filepath = self.test_dir_path / "run_inst_test.py"

        # Direct instance call with only ``solve_for`` and ``script_filepath``.
        result_path = renderer.export_script(
            solve_for="cap",
            script_filepath=script_filepath
        )

        self.assertEqual(result_path, script_filepath)
        self.assertTrue(script_filepath.exists())

    def test_export_script_instance_no_json_filepath(self):
        """Test that `export_script` raises ``ValueError`` if renderer has no ``json_filepath``."""
        renderer = object.__new__(QQTCADRenderer)
        renderer.json_filepath = None

        with self.assertRaises(ValueError) as context:
            renderer.export_script("cap")
        self.assertIn(
            "does not have `json_filepath` set",
            str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
