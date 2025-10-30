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
import re

import pyvista as pv
import shapely

from qiskit_metal import Dict, draw
from qiskit_metal.toolbox_metal.parsing import parse_entry
from qiskit_metal.renderers.renderer_base import QRendererAnalysis
from qiskit_metal.renderers.renderer_gmsh.gmsh_renderer import QGmshRenderer
from qiskit_metal.designs import MultiPlanar

QISKIT_CAPACITANCE_SCALE = 1e-15
JSON_FILENAME = "qtcad_data.json"
QTCAD_CAP_OUTPUT_FILENAME = "qtcad_output_cap.pickle"
QTCAD_EIG_OUTPUT_FILENAME = "qtcad_output_eigs.pickle"
# Template string for QTCAD’s eigenmode scalar layers.
EIGENMODE_LAYER_TEMPLATE = "abs(electric field) [V/m] - mode {n}"


def sanitize_string(string: str) -> str:
    """Remove a string’s non-alphanumeric characters/spaces/underscores/hyphens.

    Args:
        string (str): The input string to be parsed.

    Returns:
        str: The parsed string.
    """
    filename = re.sub(r'[^\w\s\-_.]', '', string)
    filename = filename.replace(' ', '_')
    return filename


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

        * capacitance_raw -- Dictionary of parameters to be passed directly to QTCAD's capacitance
                             extractor solver via `qtcad.device.capacitance.SolverParams`.
                             Overwrites the parameters defined in `capacitance`: _any_ parameters
                             from `capacitance` are ignored (see note) and the defaults from
                             `qtcad.device.capacitance.SolverParams` (see its docstring for their
                             values) take precedence, with the exception of `output_dir`,
                             `make_subdir` and `name`, which are defined by the parameters of
                             `QQTCADRenderer`.

            Note:
                For users who want full control of the solver, with the `capacitance` entry in
                `options` being enough for most applications. The keys should be the parameters
                accepted by `qtcad.device.capacitance.SolverParams`.

        * maxwell_emode -- Dictionary of parameters specific to the Maxwell eigenmode
                           extractor.

            * num_modes: Number of modes to solve for. Default: 3.
            * tol_rel: Relative tolerance on the frequency. Default: 0.05.
            * min_converged_iters -- How many consecutive iterations are required to give
                                     results that agree within the tolerance thresholds.
                                     Default: 5.

        * maxwell_emode_raw -- Dictionary of parameters to be passed directly to QTCAD's Maxwell
                               eigenmode extractor solver via
                               `qtcad.device.maxwell_eigenmode.SolverParams`.
                               Overwrites the parameters defined in `maxwell_emode`: _any_
                               parameters from `maxwell_emode` are ignored (see note) and the
                               defaults from `qtcad.device.maxwell_eigenmode.SolverParams` (see
                               its docstring for their values) take precedence, with the exception
                               of `output_dir`, `make_subdir` and `name`, which are defined by the
                               parameters of `QQTCADRenderer`.

            Note:
                For users who want full control of the solver, with the `maxwell_eigenmode` entry
                in `options` being enough for most applications. The keys should be the parameters
                accepted by `qtcad.device.maxwell_eigenmode.SolverParams`.
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
        mesh_scale=1e-3,
        materials=default_materials,
        make_subdir=True,
        capacitance=dict(
            tol_rel=0.05,
            tol_abs=0.0,
            min_converged_iters=3,
        ),
        capacitance_raw=None,
        maxwell_emode=dict(
            num_modes=3,
            tol_rel=0.05,
            min_converged_iters=5,
        ),
        maxwell_emode_raw=None,
    )
    default_junction_params = dict(
        bnd_spec = None,
        inductance = None,
        length = None,
        width = None,
        dir = None,
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
            initiate (bool): True to initiate the renderer. Defaults to `True`.
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

        # Initialize the dictionary with QTCAD-specific properties associated
        # to the tunnelling junctions of each qubit.
        self.junction_params = dict()
        for junction in self.design.qgeometry.tables["junction"].iloc:
            qubit_name = self.design._components[junction.component].name
            if qubit_name in self.junction_params:
                error_msg = ValueError(
                    "Currently, QQTCADRenderer does not support multiple"
                    " tunnelling junctions per qubit.")
                self.logger.error(error_msg)
                raise error_msg
            self.junction_params[
                qubit_name] = self.default_junction_params.copy()

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
        num_threads: int = 1,
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
            num_threads (int, optional): Number of threads for
              parallel meshing. Defaults to 1, which guarantees the meshes to
              be generated to be deterministic (for a given system
              configuration and environment) when using the Delaunay or HXT
              meshing algorithms. The latter being the recommended one.
        """

        # Minimal spacing between the vacuum box surface and the sample holder box.
        vacuum_box_min_gap = "300um"

        # Ignore the volume of metals and replace it with a list of surfaces instead
        ignore_metal_volume = True

        self.gmsh.options.mesh.min_size = initial_mesh_h_min
        self.gmsh.options.mesh.max_size = initial_mesh_h_max
        self.gmsh.options.mesh.algorithm_3d = meshing_algorithm
        # Gmsh's algorithms are deterministic when meshing using a single
        # thread, vide https://gitlab.onelab.info/gmsh/gmsh/-/issues/2255.
        self.gmsh.options.mesh.num_threads = num_threads

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

    def _current_direction(
            self,
            line: shapely.geometry.linestring.LineString) -> Union[str, None]:
        """Find the direction to be assumed for an element’s current flow.

        We assume a rectangular tunnelling junction and determine its current
        flow’s direction based on its geometry.

        Args:
            line (shapely.geometry.linestring.LineString): The `LineString`
              that defines a junction component.

        Returns:
            Union[str, None]: the current flow’s direction: `"x"`, `"y"`, with
              `None` being returned if the element is not defined along either
              the x or y directions.
        """
        # The two elements of `line.coords.xy` give us the coordinates
        # (x0, x1) and (y0, y1). When flattened and operated by `np.diff`, we
        # obtain the vector [x1-x0, y1-y0], which we can compare with unit
        # vectors to determine its direction.
        direction_vector = np.diff(line.coords.xy).flatten()

        # Unit vectors.
        unit_x = np.array([1, 0])
        unit_y = np.array([0, 1])

        # Compare inner products to distinguish the directions.
        if np.isclose(direction_vector @ unit_y, 0):
            direction = "x"
        elif np.isclose(direction_vector @ unit_x, 0):
            direction = "y"
        else:
            direction = None

        return direction

    def set_up_junction(self, qubit: str, inductance: float,
                        length: Union[float, int, str]) -> pd.DataFrame:
        """Define a qubit’s tunnelling junction (inductive port) properties.

        For eigenmode simulations, the tunnelling junction is described as an
        inductive port with linear inductance. Whilst the width is computed
        automatically from the qubit geometry, the length needs to be passed
        manually.

        Note that, for `QQTCADRenderer`, a design’s
        `qgeometry.tables['junction']` does not fully describe the properties
        of the QPU’s tunnelling junctions. One also needs to inspect
        `QQTCADRenderer.junction_params`, which this method updates.

        Args:
            qubit (str): The name of the qubit whose junction’s properties
              we will set up.
            inductance (float): The inductance (in henries) of the Josephson
              tunnelling junction approximated as a linear inductor.
            length (Union[float, int, str]): Length of the inductive port
              describing the junction. It should be the distance between
              charge islands or the gap between the island and the ground
              plane.

        Returns:
            pd.DataFrame: Table with the properties of the junction.
        """

        if qubit not in self.junction_params:
            error_msg = KeyError(f"Qubit labelled ‘{qubit}’ not found.")
            self.logger.error(error_msg)
            raise error_msg

        # The name of the physical groups associated to tunnelling junctions
        # in `QGmshRenderer` always follow the same pattern.
        junction_surface = f"{qubit}_rect_jj"

        # Access the table with the properties of the first tunnelling
        # junction of the qubit.
        junction_table = self.design.components[qubit].qgeometry_table(
            "junction")
        junction_qgeom = junction_table.iloc[0]

        # Whilst the width is easily accessible from the qubit’s geometry¹,
        # there is not a single attribute that stores information on the
        # length of junctions across all transmon-like components. For
        # instance, for `qiskit_metal.qlibrary.TransmonPocket` it would be
        # `pad_gap`, whilst for `qiskit_metal.qlibrary.TransmonCross` is
        # `cross_gap`.
        # ¹ Concerning the width, some components may have the specific
        #   property `inductor_width`, whilst other do not.
        junction_width = parse_entry(junction_qgeom.width)
        junction_length = parse_entry(length) / self.options["mesh_scale"]

        # Try to determine the direction for the inductive port’s current
        # flow.
        junction_direction = self._current_direction(junction_qgeom.geometry)
        if junction_direction is None:
            error_msg = ValueError(
                "Currently, QQTCADRenderer does not support Josephson"
                " junctions (inductive ports) not aligned along the x- or"
                " y-axes.")
            self.logger.error(error_msg)
            raise error_msg

        self.junction_params[qubit].update(
            bnd_spec=junction_surface,
            inductance=inductance,
            length=junction_length,
            width=junction_width,
            dir=junction_direction,
        )

        # Create a `pandas.DataFrame` to make easier to inspect the parsed
        # properties of the junction.
        junction_params_df = pd.DataFrame.from_dict({
            key: [value] for key, value in self.junction_params[qubit].items()
        })

        return junction_params_df

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

    def _validate_options(self):
        if not isinstance(self._options["capacitance_raw"], (type(None), dict)):
            error_msg = TypeError("`QQTCADRenderer.capacitance_raw` should either be a dictionary"
                                  " with the parameters allowed by"
                                  " `qtcad.device.maxwell_eigenmode.SolverParams` or left unset."
                                  " Please check the `options` parameters used to instantiate the"
                                  " QTCAD renderer.")
            self.logger.error(error_msg)
            raise error_msg
        if isinstance(self._options["capacitance_raw"], dict):
            warning_msg = (
                "In the renderer's `options` argument, `capacitance_raw` was defined."
                " Any capacitance solver parameters defined using the standard"
                " `capacitance` key are going to be overwritten by `capacitance_raw` and its"
                " defaults.")
            self.logger.warning(warning_msg)

        if not isinstance(self._options["maxwell_emode_raw"], (type(None), dict)):
            error_msg = TypeError("`QQTCADRenderer.maxwell_emode_raw` should either be a"
                                  " dictionary with the parameters allowed by"
                                  " `qtcad.device.maxwell_eigenmode.SolverParams` or left unset."
                                  " Please check the `options` parameters used to instantiate the"
                                  " QTCAD renderer.")
            self.logger.error(error_msg)
            raise error_msg
        if isinstance(self._options["maxwell_emode_raw"], dict):
            warning_msg = (
                "In the renderer's `options` argument, `maxwell_emode_raw` was defined."
                " Any Maxwell eigenmode solver parameters defined using the standard"
                " `maxwell_emode` key are going to be overwritten by `maxwell_emode_raw` and its"
                " defaults.")
            self.logger.warning(warning_msg)

    def export_parameters(self, json_filepath=None):
        """Exports parameters that are required for QTCAD simulations as a JSON file."""

        # Verify if the simulation parameters are valid.
        self._validate_options()

        if json_filepath is None:
            json_filepath = Path(self._options["output_dir"]) / JSON_FILENAME

        self.json_filepath = json_filepath

        data = {
            "gmsh_physical_groups": self.gmsh.physical_groups,
            "_conductors": self.conductors,
            "_inductive_ports": self.junction_params,
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
                conda_cmd, "run", "--no-capture-output", "-n", env_name, "python", "-u",
                qtcad_wrapper_path, solve_for, json_filepath
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Ensures output is a string, not a byte object.
            text=True,
        )

        # Stream the output, showing it line by line and avoiding newlines.
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

    def plot_eigenmodes(
        self,
        vtu_file: str,
        num_modes: Union[int, None] = None,
        cmap: str = "magma",
        log: bool = True,
        show: bool = True,
        save: bool = False,
    ) -> Union[str, None]:
        """Plot a grid with z=0 slices of all the eigenmodes stored in a VTU file.

        PyVista is used to generate the plots of the absolute value of the electric
        field associated to different Maxwell eigenmodes.

        Args:
            vtu_file (str): Path to the VTU file containing QTCAD’s electromagnetic
                fields.
            num_modes (Union[int, None], optional): The number of modes to plot,
                starting from the first one. Defaults to `None`, which plots all modes.
                However, if loading a VTU file from a different simulation, it must be
                set accordingly.
            cmap (str, optional): Name of the colour map to be used. Must be a colour
                map supported by PyVista. Defaults to `"magma"`.
            log (bool, optional): Whether to use a logarithmic scale when mapping data
                to colours. Defaults to `True`.
            show (bool, optional): Whether to display the plot of the electric fields.
                Defaults to `True`.
            save (bool, optional): Whether to save the plot of the electric fields as a
                PNG file. Defaults to `True`.

        Returns:
            Union[str, None]: `str` with the path to the exported image file if `save`
                was enabled. Otherwise, `None`.
        """

        if num_modes is None:
            if self._options.maxwell_emode_raw is None:
                num_modes = self._options.maxwell_emode["num_modes"]
            else:
                num_modes = self._options.maxwell_emode_raw["num_modes"]

        input_file_path = Path(vtu_file)
        output_file = None

        # Selectively load the relevant arrays from the VTU file.
        reader = pv.get_reader(input_file_path)
        # Disable all arrays.
        reader.disable_all_point_arrays()
        reader.disable_all_cell_arrays()
        # Enable the eigenmode-specific arrays and ingest the file.
        # TODO: Verify if the VTU file has all the layers.
        for mdx in range(num_modes):
            reader.enable_point_array(EIGENMODE_LAYER_TEMPLATE.format(n=mdx))
        mesh = reader.read()

        # Maximum number of axes along the horizontal direction.
        num_axes_h = 2
        # Wrap the list with the indices to the eigenmodes and get the
        # resulting list’s length. This is the desired number of axes along
        # the vertical direction.
        modes_wrapped = [
            list(range(num_modes))[i:i + num_axes_h]
            for i in range(0, num_modes, num_axes_h)
        ]
        num_axes_v = len(modes_wrapped)

        # Set up the plot.
        window_size = (500 * num_axes_h, 500 * num_axes_v)
        plotter = pv.Plotter(shape=(num_axes_v, num_axes_h),
                             window_size=window_size)
        for vdx in range(num_axes_v):
            for hdx in range(len(modes_wrapped[vdx])):
                mdx = modes_wrapped[vdx][hdx]
                scalar_layer = EIGENMODE_LAYER_TEMPLATE.format(n=mdx)
                title = f"Eigenmode {mdx+1}"

                # Create slice at z=0.
                sliced_data = mesh.slice(normal='z', origin=(0, 0, 0))

                plotter.subplot(vdx, hdx)
                # Add the sliced data.
                plotter.add_mesh(
                    sliced_data,
                    scalars=scalar_layer,
                    log_scale=log,
                    cmap=cmap,
                    # Disable the scalar bar to add a customized one later.
                    show_scalar_bar=False,
                )
                plotter.add_title(title, font_size=11)
                # Add a custom scalar bar.
                plotter.add_scalar_bar(
                    title=f"|E| (V/m), mode {mdx+1}",
                    position_x=0.15,
                    position_y=0.05,
                    width=0.7,
                    height=0.1,
                    label_font_size=10,
                )
                # Make sure the x-y plane is visible.
                plotter.view_xy()

        if show:
            plotter.show()
        if save:
            output_file = input_file_path.with_suffix(".png")
            plotter.screenshot(
                output_file.resolve(),
                transparent_background=False,
            )
            print(f"Image saved to ‘{output_file.resolve()}’.")

        del mesh

        return str(output_file.resolve())

    def plot_eigenmode(
        self,
        vtu_file: str,
        n: int = 1,
        cmap: str = "magma",
        log: bool = True,
        show: bool = True,
        save: bool = False,
    ) -> Union[str, None]:
        """Plot the z=0 slice of a given eigenmode stored in a VTU file.

        PyVista is used to generate the plot of the absolute value of the electric
        field associated to the desired Maxwell eigenmode.

        Args:
            vtu_file (str): Path to the VTU file containing QTCAD’s electromagnetic
                fields.
            n (int): Index of the desired eigenmode. Indexing starts from 1, the ground
                state.
            cmap (str, optional): Name of the colour map to be used. Must be a colour
                map supported by PyVista. Defaults to `"magma"`.
            log (bool, optional): Whether to use a logarithmic scale when mapping data
                to colours. Defaults to `True`.
            show (bool, optional): Whether to display the plot of the electric field.
                Defaults to `True`.
            save (bool, optional): Whether to save the plot of the electric field as a
                PNG file. Defaults to `True`.

        Returns:
            Union[str, None]: `str` with the path to the exported image file if `save`
                was enabled. Otherwise, `None`.
        """

        input_file_path = Path(vtu_file)
        title = f"Eigenmode {n}"
        window_size = (1000, 1000)

        # Selectively load the relevant arrays from the VTU file.
        reader = pv.get_reader(input_file_path)
        # Disable all arrays.
        reader.disable_all_point_arrays()
        reader.disable_all_cell_arrays()
        # Enable the specific eigenmode array and ingest the file.
        mdx = n - 1
        scalar_layer = EIGENMODE_LAYER_TEMPLATE.format(n=mdx)
        reader.enable_point_array(scalar_layer)
        mesh = reader.read()

        # Create slice at z=0.
        sliced_data = mesh.slice(normal='z', origin=(0, 0, 0))

        # Set up the plot.
        plotter = pv.Plotter(window_size=window_size)
        # Add the sliced data.
        plotter.add_mesh(
            sliced_data,
            scalars=scalar_layer,
            log_scale=log,
            cmap=cmap,
            # Disable the scalar bar to add a customized one later.
            show_scalar_bar=False,
        )
        plotter.add_title(title, font_size=11)
        # Add a custom scalar bar.
        plotter.add_scalar_bar(
            title=f"|E| (V/m), mode {mdx+1}",
            position_x=0.15,
            position_y=0.05,
            width=0.7,
            height=0.1,
            label_font_size=10,
        )
        # Make sure the x-y plane is visible.
        plotter.view_xy()

        if show:
            plotter.show()
        if save:
            sanitized_layer_name = sanitize_string(
                EIGENMODE_LAYER_TEMPLATE.format(n=mdx + 1))
            output_file = input_file_path.with_name(
                f"{input_file_path.stem}-{sanitized_layer_name}.png")
            plotter.screenshot(
                output_file.resolve(),
                window_size=window_size,
                transparent_background=False,
            )
            print(
                f"Image of the eigenmode {n} saved to ‘{output_file.resolve()}’."
            )

        del mesh

        return str(output_file.resolve())
