import json
from pathlib import Path
from typing import Union, Optional
import numpy as np
import sys
import logging
import pickle

logging.basicConfig()
logger = logging.getLogger("qtcad")

from qtcad.device import Device as qtcad_device
from qtcad.device.mesh3d import Mesh as qtcad_mesh
from device import materials as qtcad_materials
from qtcad.device.capacitance import Solver as QtcadSolverCap
from qtcad.device.capacitance import SolverParams as QtcadSolverCapParams
from qtcad.device.maxwell_eigenmode import Solver as QtcadSolverEig
from qtcad.device.maxwell_eigenmode import SolverParams as QtcadSolverEigParams

from dataclasses import dataclass

QTCAD_MIN_SUPPORTED_VERSION = "1.5.0"
QISKIT_CAPACITANCE_SCALE = 1e-15
QISKIT_NUMBER_EIGENMODES = 3
QTCAD_CAP_OUTPUT = "qtcad_output_cap.pickle"
QTCAD_EIG_OUTPUT = "qtcad_output_eigs.pickle"


@dataclass
class QTCADInputParams:

    gmsh_physical_groups: dict
    nets: dict
    sample_holder: bool
    qtcad_options: dict


class QQTCADWrapper():

    name = "qtcad"

    def __init__(self, json_data=None):
        """
        Args:
            json_data : Parameters imported from QQTCADRenderer.
        """

        if json_data == None:
            json_data = "qtcad_data.json"

        self.json_data = json_data
        self.gmsh_physical_groups = None
        self.nets = None
        self.sample_holder = None
        self.qtcad_options = None

    def load_and_validate(self):

        with open(self.json_data) as f:
            data = json.load(f)

        validated = QTCADInputParams(**data)
        self.gmsh_physical_groups = validated.gmsh_physical_groups
        self.nets = validated.nets
        self.sample_holder = validated.sample_holder
        self._options = validated.qtcad_options

        self._options["materials"] = dict(substrate=qtcad_materials.Si,)

    def setup(self, bnd_conditions) -> None:
        """Set up QTCAD.

        Args:
            bnd_conditions : Boundary conditions assigned on conductors.
              Default: "Dirichlet"
        """

        # Load the mesh.
        self.mesh_filepath = Path(self._options["mesh_filepath"])
        self.mesh = qtcad_mesh(self._options["mesh_scale"],
                               str(self.mesh_filepath))
        self.output_dir = Path(self.mesh_filepath.parent)

        # Instantiate the device.
        self.device = qtcad_device(self.mesh)

        # Set up dielectric volumes.
        self.dielectric_volumes = self.set_up_dielectric_volumes()

        # Set up boundary conditions
        self.set_up_boundary_conditions(bnd_conditions)

    def set_up_dielectric_volumes(
        self,
        materials: Optional[dict[str, qtcad_materials.Material]] = None,
    ) -> dict[str, qtcad_materials.Material]:
        """Define or update the properties of the model’s dielectric volumes.
        If `materials` is not given, read from the materials set up when instantiating.

        Args:
            materials (dict[str, qtcad_materials.Material], optional): Materials
              used by bodies in the design.

        Returns:
            dict[str, qtcad_materials.Material]: Information about the
              dielectric volumes in the design.
        """
        media = dict()
        if materials is None:
            materials_dict = self._options["materials"]
            if self.sample_holder:
                materials_dict["vacuum"] = qtcad_materials.vacuum
        else:
            if self.sample_holder and "vacuum" not in materials:
                logger.warn(
                    "No material properties assigned to the sample holder "
                    "(the vacuum box). We will assume it to be a vacuum. "
                    "If this is not desired, please add an entry labelled "
                    "`vacuum` to the dictionary of material properties "
                    "passed to this method.")
            materials_dict = materials

        for label, material in materials_dict.items():
            if label == "vacuum":
                medium_label = "vacuum_box"
                media[medium_label] = material
            elif label == "substrate":
                for layer, ph_geoms in self.gmsh_physical_groups.items():
                    if layer in ["global", "chips"]:
                        continue
                    for k in ph_geoms.keys():
                        # We are interested only in the dielectric volumes.
                        if not (("dielectric" in k) and ("_sfs" not in k)):
                            continue
                        medium_label = k
                        media[medium_label] = material
            else:
                # TODO Raise a warning?
                pass

        # Set properties of the dielectric media.
        for medium, material_properties in media.items():
            self.device.new_region(medium, material_properties)

        return media

    def set_up_boundary_conditions(self, bnd_conditions) -> None:
        """Assigns boundary conditions to the conductors of the device."""

        self.signal_nets = dict()
        ground_net = []
        self.conductors = []

        for layer, ph_geoms in self.gmsh_physical_groups.items():
            # ph_geoms: set of (label, Gmsh physical group ID)
            if layer in ["global", "chips"]:
                continue

            for name in ph_geoms:
                if "dielectric" in name:
                    continue
                if ("ground_plane" in name) and ("sfs" in name):
                    ground_net.append(name)

        for net, geom_names in self.nets.items():
            signal_net_surfaces = [f"{name}_sfs" for name in geom_names]

            if net == "gnd":
                ground_net += signal_net_surfaces
            else:
                self.signal_nets[geom_names[-1]] = signal_net_surfaces

        if len(self.signal_nets.keys()) == 0:
            logger.warning(
                "WARNING: No conductors were assigned to a signal net."
                "QTCAD will not be able to extract the capacitance matrix for"
                "this design.")

        # Qiskit Metal assumes the presence of the ground(_plane) conductor in
        # the capacitance matrix. Let us include the ground net as a regular
        # signal net.
        self.signal_nets["ground_plane"] = ground_net

        for net in self.signal_nets.values():
            self.conductors += net

        # Set (Dirichlet) boundary conditions for the conductors. Assign 0 V
        # to all as we are interest only in the capacitance matrix. For Maxwell
        # eigenmodes, we have perfect electric conductor (PEC) conditions.
        if bnd_conditions == "Dirichlet":
            for boundary in self.conductors:
                self.device.new_dirichlet_bnd(boundary, 0)
        elif bnd_conditions == "PEC":
            for boundary in self.conductors:
                self.device.new_pec_bnd(boundary)

        # FIXME Whilst we do not have a way of setting infinity boundary
        # conditions, let us not touch on `vacuum_box_sfs`, which then implies
        # it is a natural boundary
        # self.device.new_infinity_bnd("vacuum_box_sfs")

    def solve_eigs(self, num_modes):
        """Solve for Maxwell eigenmodes using QTCAD.

        Args:
            num_modes (int): Number of modes to solve for.

        Returns:
            The frequencies are stored in `device.maxwell_freqs'. The fields are
            also stored in `device' and can be retrieved by calling the
            corresponding field finding methods such as `device.e_field'.
        """

        # Instantiate SolverParams and load common attributes from the `_options' attribute.
        solver_params_eig = QtcadSolverEigParams()
        solver_params_eig.tol_adaptive = self._options["adaptive_tol"]
        solver_params_eig.min_converged_iters = self._options[
            "min_converged_iters"]
        solver_params_eig.output_dir = self._options["output_dir"]
        solver_params_eig.make_subdir = self._options["make_subdir"]

        # Specific attributes.
        solver_params_eig.num_modes = num_modes
        # TODO Remove hardcoded frequency before integration.
        solver_params_eig.min_freq = 1e9

        # Parse parameters.
        self.solver_params_eig = solver_params_eig
        qtcad_solver_eigs = QtcadSolverEig(self.solver_params_eig)
        self.solver_eig = qtcad_solver_eigs

        # Solve.
        eigs_out = qtcad_solver_eigs.solve(
            dvc=self.device,
            geo_file=self._options["geo_filepath"],
            name=self.name)

    def get_energy_e(self):
        """Finds the total electric energy in the device from the electric field.

        Returns:
            Electric energy given by

            .. math::
                \\int_{\\Omega} \\varepsilon \\dfrac{\\vec E^* \\cdot \\vec E}{2} d\\Omega
        """

        if hasattr(self.device, "e_field") and hasattr(self, "solver_eig"):
            e_field = self.device.e_field()
            return self.solver_eig.energy_e(self.device, e_field)

        raise ValueError("Device has no method `e_field' and/or `solver_eig'."
                         "Try calling `solve_eigs' before.")

    def get_energy_e_elems(self, mode: int):
        """Finds the electric energy in each element from the electric field.

        Args:
            mode : the electric energy in which mode.

        Returns:
            1D array representing the magnetic energy, given by

            .. math::
                \\int_{\\Omega_e} \\varepsilon \\dfrac{\\vec E^* \\cdot \\vec E}{2} d\\Omega_e

            Indices run over the elements.
        """

        if hasattr(self.device, "e_field") and hasattr(self, "solver_eig"):
            e_field = self.device.e_field()
            return self.solver_eig.energy_e_elems(self.device, e_field[...,
                                                                       mode])

        raise ValueError("Device has no method `e_field' and/or `solver_eig'."
                         "Try calling `solve_eigs' before.")

    def get_energy_b(self):
        """Finds the total magnetic energy in the device from the magnetic flux
        density.

        Returns:
            Magnetic energy given by

            .. math::
                \\int_{\\Omega} \\dfrac{\\vec B^* \\cdot \\vec B}{2 \\mu_0} d\\Omega
        """

        if hasattr(self.device, "b_field") and hasattr(self, "solver_eig"):
            b_field = self.device.b_field()
            return self.solver_eig.energy_b(self.device, b_field)

        raise ValueError("Device has no method `e_field' and/or `solver_eig'."
                         "Try calling `solve_eigs' before.")

    def get_energy_b_elems(self, mode: int):
        """Finds the magnetic energy in each element from the magnetic flux
        density.

        Args:
            mode : the electric energy in which mode.

        Returns:
            1D array representing the magnetic energy, given by

            .. math::
                \\int_{\\Omega_e} \\dfrac{\\vec B^* \\cdot \\vec B}{2 \\mu_0} d\\Omega_e

            Indices run over elements.
        """

        if hasattr(self.device, "b_field") and hasattr(self, "solver_eig"):
            b_field = self.device.b_field()
            return self.solver_eig.energy_b_elems(self.device, b_field[...,
                                                                       mode])

        raise ValueError("Device has no method `e_field' and/or `solver_eig'."
                         "Try calling `solve_eigs' before.")

    def solve_cap(self,
                  display_cap_matrix: bool = False,
                  tol_abs: Optional[float] = None,
                  max_cpus: Optional[int] = None):
        """Compute the capacitance matrix using QTCAD.

        It is stored in the attribute `capacitance_matrix', being a
        dict[tuple[str,str], float].

        Args:
            display_cap_matrix (bool, optional): Whether or not to return the
              capacitance matrix. Defaults to `False'.
            tol_abs (float, optional): Absolute tolerance. Default : `None' (use
              QTCAD's default)
            max_cpus (int, optional): The largest number of CPUs to use during
              the solution. Default: QTCAD's default, the number of logical CPUs
              available.
        """

        # Instantiate SolverParams and load common attributes from the
        # `_options' attribute.
        solver_params_cap = QtcadSolverCapParams()
        solver_params_cap.tol_rel = self._options["adaptive_tol"]
        solver_params_cap.min_converged_iters = self._options[
            "min_converged_iters"]
        solver_params_cap.output_dir = self._options["output_dir"]
        solver_params_cap.make_subdir = self._options["make_subdir"]

        if tol_abs is not None:
            solver_params_cap.tol_abs = tol_abs

        if max_cpus is not None:
            solver_params_cap.max_cpus = max_cpus

        self.solver_params_cap = solver_params_cap

        logger.info("Running QTCAD capacitance extraction.")
        self.capacitance_matrix = self.compute_capacitance_matrix()

    def save_capacitance_matrix(self, path: str) -> None:
        """Saves capacitance matrix to file.

        Args:
            path (str): path where capacitance matrix should be saved.
        """
        self.capacitance_matrix.to_csv(path, sep=" ", header=True)

    def _capacitance_matrix_to_array(self, cap) -> np.ndarray:
        sig_net_names = np.array(
            list(dict.fromkeys([k[0] for k in cap.keys()]).keys()))
        sig_net_length = len(sig_net_names)

        # Create ordered capacitance matrix.
        cap_list = [cap[(i, j)] for i in sig_net_names for j in sig_net_names]
        cap_matrix_array = np.reshape(cap_list,
                                      (sig_net_length, sig_net_length))
        return cap_matrix_array

    def compute_capacitance_matrix(self) -> dict[tuple[str, str], float]:
        """Get the capacitance matrix in femtofarads, Qiskit Metal’s default.

        Returns:
             dict: Capacitance matrix in femtofarads between the conductors.
        """

        qtcad_solver = QtcadSolverCap(self.solver_params_cap)

        # dict[tuple[str,str], float]
        cap_out = qtcad_solver.solve(dvc=self.device,
                                     signal_nets=self.signal_nets,
                                     geo_file=self._options["geo_filepath"],
                                     name=self.name)

        for cap in cap_out.keys():
            # Scale capacitance to femtofarads.
            cap_out[cap] /= QISKIT_CAPACITANCE_SCALE
            # Convert np.float64 to float to avoid issues with pickle files.
            # See https://github.com/numpy/numpy/issues/24844
            # FIXME: When Qiskit Metal updates to Numpy >2, this can be removed.
            cap_out[cap] = float(cap_out[cap])

        return cap_out


def main_solve_cap(json_data):

    qtcad_wrapper = QQTCADWrapper(json_data=json_data)
    qtcad_wrapper.load_and_validate()
    qtcad_wrapper.setup("Dirichlet")

    qtcad_wrapper.solve_cap()
    # Save results as a pickle file to be ingested by the QQTCADRenderer.
    with open(QTCAD_CAP_OUTPUT, 'wb') as file_handle:
        pickle.dump(qtcad_wrapper.capacitance_matrix,
                    file_handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def main_solve_eigs(json_data, num_modes=QISKIT_NUMBER_EIGENMODES):

    qtcad_wrapper = QQTCADWrapper(json_data=json_data)
    qtcad_wrapper.load_and_validate()
    qtcad_wrapper.setup("PEC")

    qtcad_wrapper.solve_eigs(num_modes=num_modes)

    with open(QTCAD_EIG_OUTPUT, 'wb') as file_handle:
        pickle.dump(qtcad_wrapper.device.maxwell_freqs,
                    file_handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # TODO: Move this to QQTCADRenderer.
    print("Frequencies:")
    for m in range(num_modes):
        print("mode %d: %.3f GHz" %
              (m, qtcad_wrapper.device.maxwell_freqs[m] / 1e9))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python worker.py <func1|func2> <argument>")
        sys.exit(1)

    solve_for = sys.argv[1]
    json_data = sys.argv[2]

    if solve_for == "cap":
        main_solve_cap(json_data)
    elif solve_for == "eigs":
        main_solve_eigs(json_data)
    else:
        print(f"Wrong `solve_for' argument: {solve_for}."
              "Use `cap' for capacitance or `eigs' for Maxwell eigenvalues.")
