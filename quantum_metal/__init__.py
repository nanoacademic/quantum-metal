"""Quantum Metal alias package.

This package acts as a dynamic alias for `qiskit_metal` until the rebranding is
completed.
"""

import sys
import importlib
from importlib.machinery import ModuleSpec

import qiskit_metal

# Expose `qiskit_metal` attributes in this package
globals().update(qiskit_metal.__dict__)

# Alias the top-level package in `sys.modules`.
sys.modules["quantum_metal"] = qiskit_metal

# Custom loader to return the exact same module object from `sys.modules`.
class AliasLoader:

    def create_module(self, spec):
        real_name = spec.name.replace("quantum_metal", "qiskit_metal", 1)
        return sys.modules[real_name]

    def exec_module(self, module):
        pass


# Custom finder to map `quantum_metal` submodules to `qiskit_metal` submodules.
class QuantumMetalFinder:

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("quantum_metal."):
            real_name = fullname.replace(
                "quantum_metal.", "qiskit_metal.", 1
            )
            try:
                mod = importlib.import_module(real_name)
                is_pkg = hasattr(mod, "__path__")
                return ModuleSpec(fullname, AliasLoader(), is_package=is_pkg)
            except ImportError:
                return None
        return None


# Register the finder at the beginning of `sys.meta_path`.
sys.meta_path.insert(0, QuantumMetalFinder())
