from dataclasses import dataclass


@dataclass
class QtcadInputParams:
    gmsh_physical_groups: dict
    _conductors: dict
    _inductive_ports: dict
    sample_holder: bool
    qtcad_options: dict
    cap_refined_mesh_file: str
    eig_refined_mesh_file: str
    eig_field_file: str


class QtcadConstants:
    DEFAULT_JSON_FILENAME = "qtcad_data.json"
    """Default file name for `QQTCADRenderer`’s associated JSON file."""

    QISKIT_CAPACITANCE_SCALE = 1e-15
    """Capacitance scale/units in Qiskit Metal"""

    QTCAD_CAP_OUTPUT_FILENAME = "qtcad_output_cap.pickle"
    """Default file name for the pickle file with capacitance results."""

    QTCAD_EIG_OUTPUT_FILENAME = "qtcad_output_eigs.pickle"
    """Default file name for the pickle file with Maxwell eigenmode results."""

    EIGENMODE_LAYER_TEMPLATE = "abs(electric field) [V/m] - mode {n}"
    """Template string for QTCAD®’s Maxwell eigenmode scalar layers in VTU files."""
