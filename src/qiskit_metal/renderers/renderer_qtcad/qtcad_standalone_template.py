"""QTCAD standalone script generation template."""


def get_standalone_script(
    solve_for,
    mesh_filepath,
    mesh_scale,
    dielectric_regions_code,
    signal_conductors,
    conductors,
    inductive_ports_code,
    capacitance_raw,
    cap_tol_rel,
    cap_tol_abs,
    cap_min_iters,
    output_dir,
    make_subdir,
    geo_filepath,
    maxwell_emode_raw,
    eig_num_modes,
    eig_tol_rel,
    eig_min_iters,
) -> str:
    """Returns a standalone QTCAD simulation Python script as a string."""
    common_header = f"""import pickle
from pathlib import Path
from time import time

from qtcad.device import Device as qtcad_device
from qtcad.device.mesh3d import Mesh as qtcad_mesh
from qtcad.device import materials as qtcad_materials

# Set up mesh and `Device`.
mesh_filepath = Path({repr(str(mesh_filepath))})
mesh_scale = {mesh_scale}
mesh = qtcad_mesh(mesh_scale, str(mesh_filepath))
device = qtcad_device(mesh)

# Set up materials.
{dielectric_regions_code}

# Set up boundary conditions.
conductors = {repr(conductors)}"""

    if solve_for == "cap":
        return f"""# Standalone QTCAD capacitance simulation script from Quantum Metal.

{common_header}
signal_conductors = {repr(signal_conductors)}

from qtcad.device.capacitance import Solver as SolverCap
from qtcad.device.capacitance import SolverParams as SolverCapParams

for boundary in conductors:
    device.new_dirichlet_bnd(boundary, 0)

# Inductive ports
{inductive_ports_code}

# Set up capacitance solver.
options_cap_raw = {repr(capacitance_raw)}
if options_cap_raw is None:
    solver_params_cap = SolverCapParams()
    solver_params_cap.tol_rel = {cap_tol_rel}
    solver_params_cap.tol_abs = {cap_tol_abs}
    solver_params_cap.min_converged_iters = {cap_min_iters}
else:
    solver_params_cap = SolverCapParams(options_cap_raw)

solver_params_cap.output_dir = {repr(output_dir)}
solver_params_cap.make_subdir = {make_subdir}
solver_params_cap.name = "qtcad"

qtcad_solver = SolverCap(
    device,
    signal_conductors,
    solver_params_cap,
    geo_file={repr(geo_filepath)}
)

# Run the simulation and output results.
print("Running capacitance-matrix solver...")
t0 = time()
capacitance_matrix = qtcad_solver.solve()
dt = time() - t0
print("Solution completed in %.2f s" % dt)

# Display results.
qtcad_solver.print_cap(capacitance_matrix, decimals=3)

# Save results as a pickle file.
out_dir = Path({repr(output_dir)})
out_dir.resolve().mkdir(exist_ok=True, parents=True)
filepath = out_dir / "qtcad_output_cap.pickle"
with open(filepath, 'wb') as f:
    pickle.dump(
        capacitance_matrix, f, protocol=pickle.HIGHEST_PROTOCOL
    )
print(f"Capacitance matrix saved to {{filepath}}")
"""
    elif solve_for == "eigs":
        return f"""# Standalone QTCAD Maxwell eigenmode simulation script from Quantum Metal.

{common_header}

from qtcad.device.maxwell_eigenmode import Solver as SolverEig
from qtcad.device.maxwell_eigenmode import SolverParams as SolverEigParams

for boundary in conductors:
    device.new_pec_bnd(boundary)

# Inductive ports.
{inductive_ports_code}

# Set up Maxwell-eigenmode solver.
options_maxwell_emode_raw = {repr(maxwell_emode_raw)}
if options_maxwell_emode_raw is None:
    solver_params_eig = SolverEigParams()
    solver_params_eig.num_modes = {eig_num_modes}
    solver_params_eig.tol_rel = {eig_tol_rel}
    solver_params_eig.min_converged_iters = {eig_min_iters}
else:
    solver_params_eig = SolverEigParams(options_maxwell_emode_raw)

solver_params_eig.output_dir = {repr(output_dir)}
solver_params_eig.make_subdir = {make_subdir}
solver_params_eig.name = "qtcad"

qtcad_solver_eigs = SolverEig(
    device, solver_params_eig, geo_file={repr(geo_filepath)}
)

# Run the simulation and output results.
print("Running Maxwell-eigenmodes solver...")
t0 = time()
qtcad_solver_eigs.solve()
dt = time() - t0
print("Solution completed in %.2f s" % dt)

# Display results.
for i, freq in enumerate(device.maxwell_freqs):
    freq_ghz = freq / 1e9
    print(f"Frequency of mode {{i}}: {{freq_ghz:.3f}} GHz")

# Save results as a pickle file.
out_dir = Path({repr(output_dir)})
out_dir.resolve().mkdir(exist_ok=True, parents=True)
filepath = out_dir / "qtcad_output_eigs.pickle"
with open(filepath, 'wb') as f:
    pickle.dump(
        device.maxwell_freqs.tolist(),
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )
print(f"Maxwell eigenmode frequencies saved to {{filepath}}")
"""
    else:
        raise ValueError(f"Unknown solve_for: '{solve_for}'. Must be 'cap' or 'eigs'.")
