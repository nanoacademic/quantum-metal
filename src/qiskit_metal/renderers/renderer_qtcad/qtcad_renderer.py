# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Union, Optional
import shutil
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import json
import pickle

from qiskit_metal import Dict, draw
from qiskit_metal.renderers.renderer_base import QRendererAnalysis
from qiskit_metal.renderers.renderer_gmsh.gmsh_renderer import QGmshRenderer
from qiskit_metal.designs import MultiPlanar

QISKIT_CAPACITANCE_SCALE = 1e-15
JSON_FILENAME = "qtcad_data.json"
QTCAD_CAP_OUTPUT_FILENAME = "qtcad_output_cap.pickle"
QTCAD_EIG_OUTPUT_FILENAME = "qtcad_output_eigs.pickle"



class QQTCADRenderer(QRendererAnalysis):
    """Extends QRendererAnalysis class to use QTCAD’s API with Gmsh’s meshes.
    Based on QElmerRenderer.

    QQTCADRenderer default options:
        * adaptive -- Adaptive mesh refinement flag. Defaults to `True`.
        * adaptive_mesh_scale -- Scale factor of the design files. Default: 1e-3,
                                 millimetre.
        * output_dir -- Directory that will be used to store the
                        refined meshes and auxiliary files. Defaults to the current
                        directory.
        * geo_filepath -- Path to the design file containing the geometry of the
                          problem (recommended format: .xao).
                          Default: `qiskit_device.xao`
        * mesh_filepath -- Path to the mesh file (recommended format: .msh4).
                           Default: `qiskit_device.msh4`
        * mesh_scale -- Scaling factor of the static meshes. Default: 1e-3, millimetre.
        * materials -- Dictionary to store material properties.
        * make_subdir -- if `True`, computation results will be stored in a new
                         subdirectory within output_dir labeled by the current date and
                         time. Defaults to `True`

        * capacitance -- Dictionary of parameters specific to the capacitance extractor.

            * tol_rel: relative tolerance (see Note). Defaults to 0.05.
            * tol_abs: absolute tolerance (see Note). Defaults to 0.0.
            * min_converged_iters -- How many consecutive iterations are required to give
                                 results that agree within the tolerance thresholds.
                                 Default: 3.

            Note:
                The convergence threshold for each entry of the capacitance
                matrix C_ij is set as tol_rel * |C_{ij}| + tol_abs. The attribute
                tol_abs is useful when :math:`|C_{ij}|` is expected to be zero or
                very small.

        * maxwell_emode -- Dictionary of parameters specific to the Maxwell eigenmode
                                        extractor.

            * tol_rel: relative tolerance on the frequency. Defaults to 0.05.
            * min_converged_iters -- How many consecutive iterations are required to give
                                 results that agree within the tolerance thresholds.
                                 Default: 5.
    """

    # The defaults of a renderer must be in a dict named `default_options`. They
    # can be overwritten by passing an `options` dictionary when instantiating
    # QQTCADRenderer.
    default_materials = dict(substrate="qtcad_materials.Si",)
    default_options = dict(
        adaptive=True,
        adaptive_mesh_scale=1e-3,
        output_dir=".",
        geo_filepath="qiskit_device.xao",
        mesh_filepath="qiskit_device.msh4",
        mesh_scale=1-3,
        materials=default_materials,
        make_subdir=True,
        capacitance=dict(
            tol_rel=0.05,
            tol_abs=0.0,
            min_converged_iters=3,
        ),
        maxwell_emode=dict(
            tol_rel=0.05,
            min_converged_iters=5,
        ),
    )

    name = "qtcad"
    """name"""

    def __init__(
        self,
        design: "MultiPlanar",
        layer_types: Union[dict, None] = None,
        initiate=True,
        options: Dict = None,
    ):
        """
        Args:
            design ('MultiPlanar'): The design.
            layer_types (Union[dict, None]): The type of layer in the format:
              dict(metal=[...], dielectric=[...]). Defaults to `None`.
            initiate (bool): True to initiate the renderer. Defaults to `False`.
            options (Dict, optional): Used to override default options. Defaults
              to `None`.
        """

        default_layer_types = dict(metal=[1], dielectric=[3])
        self.layer_types = (default_layer_types
                            if layer_types is None else layer_types)

        super().__init__(design=design, initiate=initiate, options=options)

        self.mesh_file = None
        self.json_filepath = None

        # If not using adaptive meshing, set adaptive parameters to None.
        # FIXME In the future, it would be desirable to have a flag in the
        # capacitance routine that determines if we want or not to use adaptive
        # meshing without having to set all the arguments associated to it to
        # None.
        if self._options["adaptive"] == False:
            for arg in [
                    "adaptive_mesh_scale",
                    "geo_filepath",
            ]:
                self._options[arg] = None

    @property
    def initialized(self):
        """Check if the renderer is ready to be used.

        Must return `True` if successful, `False` otherwise.
        """
        status = all([self.gmsh.initialized, self._qtcad_ready])
        return status

    def initialize_renderer(self):
        """Initializes the Gmsh and QTCAD renderers.

        NOTE TO THE USER: this should be used when using the QQTCADRenderer
        through design.renderers.qtcad instance.

        Example: To change one of the options exposed by QGmshRenderer, one has
        to first initiate the QQTCADRenderer using this method and then change
        the property like so:

        ```
        design.renderers.qtcad.initialize_renderer()
        design.renderers.qtcad.gmsh.options.mesh.num_threads = 1
        ...
        ```

        """
        self.gmsh = QGmshRenderer(self.design, self.layer_types)
        self._qtcad_ready = self.check_environment()

    def _initiate_renderer(self):
        """Initializes the Gmsh renderer and check QTCAD requirements.

        NOTE: This is automatically called when the USER specifically imports
        QQTCADRenderer in a Jupyter notebook and instantiates it.
        """
        self.gmsh = QGmshRenderer(self.design, self.layer_types)
        self._qtcad_ready = self.check_environment()
        status = all([self.gmsh.initialized, self._qtcad_ready])
        return status

    def _close_renderer(self):
        """Finalizes the Gmsh renderer."""
        self.gmsh.close()

    def close(self):
        """Public method to close the Gmsh renderer."""
        return self._close_renderer()

    def check_environment(self) -> bool:
        """Check if QTCAD is able to fully run as a renderer."""

        # ParaView has a Python API, but usually it is not integrated into
        # users’ Python environment.
        self._check_paraview = shutil.which("paraview") is not None
        if not self._check_paraview:
            self.logger.warning(
                "ParaView was not found in the user’s path."
                " Please install it if you want post-processing visualization.")

        return True

    def render_design(
        self,
        selection: Union[list, None] = None,
        open_pins: Union[list, None] = None,
        box_plus_buffer: bool = True,
        draw_sample_holder: bool = True,
        skip_junctions: bool = True,
        mesh_geoms: bool = True,
        omit_ground_for_layers: Optional[list[int]] = None,
        initial_mesh_h_min: str = "150um",
        initial_mesh_h_max: str = "150um",
        meshing_algorithm: int = 10,
    ):
        """Render the design in Gmsh and apply changes to modify the geometries
        according to the type of simulation. Simulation parameters provided by
        the user.

        Args:
            selection (Union[list, None], optional): List of selected components
              to render. Defaults to `None`.
            open_pins (Union[list, None], optional): List of open pins that are
              open. Defaults to `None`.
            box_plus_buffer (bool, optional): Set to `True` for adding buffer to
              chip dimensions. Defaults to `True`.
            draw_sample_holder (bool, optional): Set to `True` to draw the sample
              holder box. Defaults to `True`.
            skip_junctions (bool, optional): Set it to `True` to skip rendering the
              junctions. Defaults to `False`.
            mesh_geoms (bool, optional): Set to `True` for meshing the geometries.
              Defaults to `True`.
            omit_ground_for_layers (Optional[list[int]]): Omit rendering the
              ground plane for specified layers. Defaults to `None`.
            initial_mesh_h_min (str, optional): Minimum caracteristic length for
              the first mesh (adaptive) or for the only mesh (static).
            initial_mesh_h_max (str, optional): Maximum caracteristic length for
              the first mesh (adaptive) or for the only mesh (static).
            meshing_algorithm (int, optional): Gmsh's 3D mesh algorithm. The
              possible values are 1 (Delaunay), 3 (initial mesh only),
              4 (frontal), 7 (MMG3D), 9 (R-tree), 10 (HXT). (Default: 10)
        """

        # Minimal spacing between the vacuum box surface and the sample holder box.
        vacuum_box_min_gap = "300um"

        # Ignore the volume of metals and replace it with a list of surfaces instead
        ignore_metal_volume = True

        self.gmsh.options.mesh.min_size = initial_mesh_h_min
        self.gmsh.options.mesh.max_size = initial_mesh_h_max
        self.gmsh.options.mesh.algorithm_3d = meshing_algorithm

        # For handling the case when the user wants to use
        # QQTCADRenderer from design.renderers.qtcad instance.
        if not self.initialized:
            status = self._initiate_renderer()
            if not status:
                self.logger.warning(
                    "Unable to initialise QQTCADRenderer."
                    " Please check and address any warning and error message.")

        self.gmsh.render_design(
            selection=selection,
            open_pins=open_pins,
            box_plus_buffer=box_plus_buffer,
            draw_sample_holder=draw_sample_holder,
            skip_junctions=skip_junctions,
            mesh_geoms=mesh_geoms,
            ignore_metal_volume=ignore_metal_volume,
            omit_ground_for_layers=omit_ground_for_layers,
            vacuum_box_min_gap=vacuum_box_min_gap,
        )

        self.sample_holder = draw_sample_holder
        self.qcomp_geom_table = self.get_qgeometry_table()
        self.conductors = self.assign_conductors(open_pins=open_pins)

    def get_qgeometry_table(self) -> pd.DataFrame:
        """Combines the `path` and `poly` qgeometry tables into a single table,
        and adds column containing the minimum z coordinate of the layer
        associated with each qgeometry.

        Raises:
            ValueError: Raised when component in selection of qcomponents is not
              in QDesign.

        Returns:
            pd.DataFrame: Table with elements in both "path" and "poly"
              qgeometry tables.
        """

        if self.gmsh.case == 0:
            qcomp_ids = self.gmsh.qcomp_ids
        elif self.gmsh.case == 1:
            qcomp_ids = list(set(self.design._components.keys()))
        elif self.gmsh.case == 2:
            raise ValueError("Selection provided is invalid.")

        metal_layers = self.layer_types["metal"]

        mask = (lambda table: table["component"].isin(qcomp_ids) & ~table[
            "subtract"] & table["layer"].isin(metal_layers))

        min_z = lambda layer: min(
            sum(self.gmsh.get_thickness_zcoord_for_layer_datatype(layer)),
            self.gmsh.get_thickness_zcoord_for_layer_datatype(layer)[1],
        )

        path_table = self.design.qgeometry.tables["path"]
        poly_table = self.design.qgeometry.tables["poly"]
        qcomp_paths = path_table[mask(table=path_table)]
        qcomp_polys = poly_table[mask(table=poly_table)]
        qcomp_geom_table = pd.concat([qcomp_paths, qcomp_polys],
                                     ignore_index=True)

        qcomp_geom_table["min_z"] = qcomp_geom_table["layer"].apply(min_z)
        qcomp_geom_table.sort_values(by=["min_z"],
                                     inplace=True,
                                     ignore_index=True)

        return qcomp_geom_table

    def assign_conductors(
        self,
        open_pins: Union[list,
                         None] = None) -> dict[Union[str, int], list[str]]:
        """Assigns a netlist number to each galvanically connected metal region (a
        ‘signal conductor’) and returns a dictionary with each net as a key, and the
        corresponding list of geometries associated with that net (conductor) as values.

        Args:
            open_pins (Union[list, None], optional): List of tuples of pins that
              are open. Defaults to None.

        Returns:
            dict[Union[str, int], list[str]]: dictionary with keys for each net,
              and list of values with the corresponding geometries associated
              with that net as values.

        """

        netlists = dict()
        netlist_id = 0

        qgeom_names = self.qcomp_geom_table["name"]
        qcomp_names_for_qgeom = [
            list(self.design.components.keys())[i - 1]
            for i in self.qcomp_geom_table["component"]
        ]
        phys_grps = [
            s1 + "_" + s2 for s1, s2 in zip(qcomp_names_for_qgeom, qgeom_names)
        ]
        qgeom_idxs = list(range(len(self.qcomp_geom_table)))
        id_net_dict = {k: -1 for k in phys_grps}

        while len(qgeom_idxs) != 0:
            i = qgeom_idxs.pop(0)
            shape_i = self.qcomp_geom_table.iloc[[i]]["geometry"][i]
            chip_i = self.qcomp_geom_table.iloc[[i]]["chip"][i]
            layer_i = self.qcomp_geom_table.iloc[[i]]["layer"][i]
            (
                thick_i,
                z_coord_i,
            ) = self.gmsh.get_thickness_zcoord_for_layer_datatype(layer_i)
            id_net_dict[phys_grps[i]] = (netlist_id if
                                         (id_net_dict[phys_grps[i]]
                                          == -1) else id_net_dict[phys_grps[i]])
            for j in qgeom_idxs:
                shape_j = self.qcomp_geom_table.iloc[[j]]["geometry"][j]
                chip_j = self.qcomp_geom_table.iloc[[j]]["chip"][j]
                layer_j = self.qcomp_geom_table.iloc[[j]]["layer"][j]
                (
                    thick_j,
                    z_coord_j,
                ) = self.gmsh.get_thickness_zcoord_for_layer_datatype(layer_j)
                dist = shape_i.distance(shape_j)

                layers_touch = False
                if (layer_i == layer_j or z_coord_j == z_coord_i or
                        z_coord_i + thick_i == z_coord_j or
                        z_coord_j + thick_j == z_coord_i):
                    layers_touch = True

                if dist == 0.0 and chip_i == chip_j and layers_touch:
                    if id_net_dict[phys_grps[j]] == -1:
                        id_net_dict[phys_grps[j]] = id_net_dict[phys_grps[i]]
                    elif (id_net_dict[phys_grps[j]]
                          != id_net_dict[phys_grps[i]]):
                        net_id_i = id_net_dict[phys_grps[i]]
                        for k, v in id_net_dict.items():
                            if v == net_id_i:
                                id_net_dict[k] = id_net_dict[phys_grps[j]]

            if -1 not in id_net_dict.values():
                break

            netlist_id = max(list(id_net_dict.values())) + 1

        gnd_phys_grps = self.get_gnd_qgeoms(open_pins)
        gnd_netlist = list({
            net for (name, net) in id_net_dict.items()
            if (name in gnd_phys_grps)
        })

        netlists["gnd"] = list()
        for k, v in id_net_dict.items():
            if v in gnd_netlist:
                netlists["gnd"].append(k)
            else:
                if v not in netlists.keys():
                    netlists[v] = list()
                netlists[v].append(k)

        netlists = {
            (i - 1 if (k != "gnd") else k): v
            for i, (k, v) in enumerate(netlists.items())
        }

        return netlists

    def get_gnd_qgeoms(self, open_pins: Union[list, None] = None) -> list[str]:
        """Obtain a list of qgeometry names associated with pins shorted to
        ground.

        Args:
            open_pins (Union[list, None], optional): List of tuples of pins that
              are open. Defaults to None.

        Returns:
            list[str]: Names of qgeometry components with pins connected to
              ground plane.
        """

        open_pins = open_pins if open_pins is not None else []
        qcomp_lst = self.design.components.keys()
        all_pins = list()
        gnd_qgeoms = set()

        for qcomp in qcomp_lst:
            qcomp_pins = self.design.components[qcomp].pins.keys()
            all_pins += list(zip([qcomp] * len(qcomp_pins), qcomp_pins))

        nets_table = self.design.net_info
        gnd_nets = list(nets_table[nets_table["pin_name"] == "short"]["net_id"])
        gnd_nets_mask = nets_table["net_id"].isin(gnd_nets)
        port_nets_table = nets_table[~gnd_nets_mask]
        port_pins_names = port_nets_table["pin_name"]
        qcomp_names_for_port_pins = [
            list(qcomp_lst)[i - 1] for i in port_nets_table["component_id"]
        ]
        port_pins = list(zip(qcomp_names_for_port_pins, port_pins_names))
        all_open_pins = list(set(port_pins + open_pins))
        gnd_pins = [pin for pin in all_pins if pin not in all_open_pins]

        for pin in gnd_pins:
            qcomp_name, qcomp_pin = pin
            pin_qcomp_id = self.design.components[qcomp_name].id

            if pin_qcomp_id in list(self.qcomp_geom_table["component"]):
                pin_qcomp_geom_table = self.qcomp_geom_table[
                    self.qcomp_geom_table["component"] == pin_qcomp_id]
                pin_point = draw.Point(
                    self.design.components[qcomp_name].pins[qcomp_pin]
                    ["middle"])

                for _, qgeom_row in pin_qcomp_geom_table.iterrows():
                    if pin_point.intersects(qgeom_row["geometry"]):
                        gnd_qgeoms.add(qcomp_name + "_" + qgeom_row["name"])

        return list(gnd_qgeoms)

    def render_chips(
        self,
        chips: Union[str, list[str]] = [],
        draw_sample_holder: bool = True,
        box_plus_buffer: bool = True,
    ):
        """Render all chips of the design.

        Calls `render_chip` for each chip.
        """
        pass

    def render_chip(self, chip_name: str):
        """Render the given chip.

        Args:
            name (str): chip to render
        """
        pass

    def render_layers(
        self,
        draw_sample_holder: bool = True,
        layers: Union[list[int], None] = None,
        box_plus_buffer: bool = True,
    ):
        """Render all layers of the design.

        Calls `render_layer` for each layer.
        """
        self.gmsh.render_layers(draw_sample_holder, layers, box_plus_buffer)

    def render_layer(self, layer_number: str, datatype: int = 0):
        """Render the given layer.

        Args:
            name (str): layer to render
        """
        self.gmsh.render_layer(layer_number, datatype)

    def render_components(self, table_type: str):
        """Render all components of the design.

        If selection is `None`, then render all components.

        Args:
            selection (QComponent): Component to render.
        """
        self.gmsh.render_components(table_type)

    def render_component(self, component):
        """Render the specified component.

        Args:
            component (QComponent): Component to render.
        """
        self.gmsh.render_component(component)

    def render_element(self, qgeom: pd.Series, table_type: str):
        """Render the specified element

        Args:
            element (Element): Element to render.
        """
        self.gmsh.render_element(qgeom, table_type)

    def render_element_path(self, path: pd.Series):
        """Render an element path.

        Args:
            path (str): Path to render.
        """
        self.gmsh.render_element_path(path)

    def render_element_junction(self, junc: pd.Series):
        """Render an element junction.

        Args:
            junc (str): Junction to render.
        """
        self.gmsh.render_element_junction(junc)

    def render_element_poly(self, poly: pd.Series):
        """Render an element poly.

        Args:
            poly (Poly): Poly to render.
        """
        self.gmsh.render_element_poly(poly)

    def save_screenshot(self, path: str = None, show: bool = True):
        """Save a screenshot.

        Args:
            path (str, optional): Path to save the file. Defaults to None.
            show (bool, optional): Whether or not to display the screenshot.
              Defaults to True.

        Returns:
            pathlib.WindowsPath: path to the PNG file screenshot.
        """
        self.gmsh.save_screenshot(path, show)

    def launch_gmsh_gui(self):
        """Launch Gmsh GUI for viewing the model."""
        self.gmsh.launch_gui()

    def export_mesh(
        self,
        mesh_file: Optional[str] = None,
        geometry_file: Optional[str] = None,
    ):
        """Export the mesh and geometry files.

        The mesh is exported with unit scaling factor.

        Args:
            mesh_file (Optional[str], optional): File path to which to save the
              mesh file.
            geometry_file (Optional[str], optional): File path to which to save
              the geometry file.
        """

        if geometry_file is None:
            geometry_file = self._options["geo_filepath"]

        # We check against `None` once again because
        # `self._options["geo_filepath"]` is `None` if AMR is not enabled.
        if geometry_file is not None:
            Path(geometry_file).parent.resolve().mkdir(exist_ok=True, parents=True)
            self.gmsh.export_geometry(geometry_file)

        if mesh_file is None:
            mesh_file = self._options["mesh_filepath"]

        self.mesh_file = mesh_file

        # Guarantee the path to the mesh file exists.
        Path(self.mesh_file).parent.resolve().mkdir(exist_ok=True, parents=True)
        self.gmsh.export_mesh(self.mesh_file, scaling_factor=1)

    def display_post_processing_data(self, signal_conductor: str) -> None:
        """Post-process the data output by QTCAD in ParaView.

        Args:
            signal_conductor (str): Signal conductor for post-processing.
        """
        if signal_conductor not in self.signal_conductors:
            self.logger.error(
                f"No signal conductor “{self.signal_conductors}” found in the model."
                " The signal conductors defined in this model are:\n"
                ", ".join([f'"{cond}"' for cond in self.signal_conductors]))
        else:
            arguments = ["paraview", self.post_process_files[signal_conductor]]
            subprocess.call(arguments, cwd=self.output_dir)

    def export_parameters(self, json_filepath=None):
        """Exports parameters that are required for QTCAD simulations as a JSON file."""

        if json_filepath is None:
            json_filepath = Path(self._options["output_dir"]) / JSON_FILENAME

        self.json_filepath = json_filepath

        data = {
            "gmsh_physical_groups": self.gmsh.physical_groups,
            "_conductors": self.conductors,
            "sample_holder": self.sample_holder,
            "qtcad_options": {
                k: v for k, v in self._options.items() if k != "materials"
            },
        }

        # Guarantee the path to the JSON file exists.
        Path(json_filepath).parent.resolve().mkdir(exist_ok=True, parents=True)
        with open(json_filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _check_conda_env(self, env_name) -> bool:
        """Check the existence of a conda environment.

        Args:
            env_name (str): name of the conda environment to check for its existence.
        """

        conda_cmd = shutil.which("conda")
        proc = subprocess.run(
            [conda_cmd, "list", "--name", env_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return_code = proc.returncode

        if return_code == 0:
            return True
        return False

    def run_qtcad(self,
                  solve_for,
                  json_filepath=None,
                  env_name="qtcad",
                  ):
        """Calls wrapper file in a specific environment

            Args:
                solve_for (str): Solve for `cap` (capacitance matrix) or `eigs`
                  (Maxwell eigenmodes).
                json_filepath (str): Path to the necessary parameters for the solver.
                  Defaults to the value used when exporting them using
                  `export_parameters`.
                env_name (str): name of the conda environment where QTCAD is available.
                  Default: qtcad

        """
        # Path to the QTCAD wrapper routine.
        qtcad_wrapper_path = str(Path(__file__).parent.resolve() / "wrapper.py")

        if json_filepath is None:
            json_filepath = self.json_filepath

        if json_filepath is None:
            raise Exception(
                "Unable to find the JSON file with the input parameters to run"
                " QTCAD simulations."
                " Please make sure to have exported them using"
                " `export_parameters`."
                )
        if not Path(json_filepath).exists():
            raise Exception(
                "Unable to find the JSON file with the input parameters to run"
                f" QTCAD simulations at ‘{json_filepath}’."
                " Please make sure to have exported them using"
                " `export_parameters`."
                " If a custom path was provided, make sure it points to a valid"
                " QTCAD JSON file."
                )

        if (self.mesh_file is None) or (not Path(self.mesh_file).exists()):
            raise Exception(
                "Unable to find the mesh file."
                " Please make sure to have generated it using `export_mesh`."
                )

        qtcad_env_found = self._check_conda_env(env_name)
        if not qtcad_env_found:
            raise Exception(
                f"Unable to find QTCAD's conda environment ‘{env_name}’."
                " If you have installed QTCAD in a custom environment, please provide its name"
                " using the parameter `env_name`."
                )

        # Launch a subprocess with unbuffered Python (-u).
        conda_cmd = shutil.which("conda")

        self.logger.info("================")
        self.logger.info("Running QTCAD...")
        self.logger.info("================")
        process = subprocess.Popen(
            [
                conda_cmd, "run", "-n", env_name, "python", "-u", qtcad_wrapper_path,
                solve_for, json_filepath
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Ensures output is a string, not a byte object.
            text=True,
        )

        # TODO: Stream the output.
        # Show the output line by line, avoiding newlines.
        for line in process.stdout:
            print(line, end="")

        process.wait()


    def load_qtcad_capacitance_matrix(self, filepath: Union[str, None] = None) -> pd.DataFrame:
        """Load capacitance matrix from file.

        Args:
            filename (str, optional): Path to the pickle file containing QTCAD's capacitance
            matrix (a dictionary; units: femtofarads). Defaults to `None`, loading the
            default path to the file written by the capacitance extractor.

        Returns:
            pd.DataFrame: Table containing the capacitance matrix.
        """
        if filepath is None:
            filepath = Path(self._options["output_dir"]) / QTCAD_CAP_OUTPUT_FILENAME

        input_file = Path(filepath)
        if not input_file.exists():
            raise Exception(
                f"Unable to load the capacitance matrix generated by QTCAD from ‘{filepath}’."
                " Please make sure the path to the file is correct and the capacitance"
                " extraction method has ran successfully.",
                )

        with open(filepath, 'rb') as handle:
            cap = pickle.load(handle)

        # TODO Check validity of data.

        # Parse dictionary and create an ordered capacitance matrix.
        sig_conductor_names = np.array(
            list(dict.fromkeys([k[0] for k in cap.keys()]).keys()))
        sig_conductor_length = len(sig_conductor_names)
        cap_list = [cap[(i, j)] for i in sig_conductor_names for j in sig_conductor_names]
        cap_matrix_array = np.reshape(cap_list,
                                        (sig_conductor_length, sig_conductor_length))
        cap_matrix_df = pd.DataFrame(
            cap_matrix_array,
            index=sig_conductor_names,
            columns=sig_conductor_names,
        )

        return cap_matrix_df

    def load_qtcad_maxwell_eigenmodes(self, filepath: Union[str, None] = None) -> np.ndarray:
        """Load Maxwell eigenmodes from file.

        Args:
            filename (str, optional): Path to the pickle file containing QTCAD's Maxwell
            eigenmode calculation result (units: gigahertz). Defaults to `None`, loading the
            default path to the file written by the Maxwell eigenmode extractor.

        Returns:
            nd.ndarray: Ordered list of Maxwell eigenmodes.
        """
        if filepath is None:
            filepath = Path(self._options["output_dir"]) / QTCAD_EIG_OUTPUT_FILENAME

        input_file = Path(filepath)
        if not input_file.exists():
            raise Exception(
                f"Unable to load Maxwell eigenmodes generated by QTCAD from ‘{filepath}’."
                " Please make sure the path to the file is correct and the eigenmode"
                " extraction method has ran successfully."
                )

        with open(filepath, 'rb') as handle:
            eig = pickle.load(handle)

        # TODO Check validity of data.

        frequencies = pd.DataFrame(eig, columns=["Frequency (GHz)"])
        frequencies.index.name = "Eigenmode"

        return frequencies / 1e9

