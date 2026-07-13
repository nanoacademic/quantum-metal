from typing import List, Tuple, Optional
from qiskit_metal.designs.design_base import QDesign


def get_min_bounding_box(
    design: QDesign, qcomp_ids: List[int], case: int, logger
) -> Tuple[float]:
    """
    Determine the max/min x/y coordinates of the smallest rectangular, axis-aligned
    bounding box that will enclose a selection of components to render, given by
    self.qcomp_ids. This method is only used when box_plus_buffer is True.

    Returns:
        Tuple[float]: min x, min y, max x, and max y coordinates of bounding box.
    """
    min_x_main = min_y_main = float("inf")
    max_x_main = max_y_main = float("-inf")
    if case == 2:  # One or more components not in QDesign.
        logger.warning("One or more components not found.")
    elif case == 1:  # All components rendered.
        for qcomp in design.components:
            min_x, min_y, max_x, max_y = design.components[qcomp].qgeometry_bounds()
            min_x_main = min(min_x, min_x_main)
            min_y_main = min(min_y, min_y_main)
            max_x_main = max(max_x, max_x_main)
            max_y_main = max(max_y, max_y_main)
    else:  # Strict subset rendered.
        for qcomp_id in qcomp_ids:
            min_x, min_y, max_x, max_y = design._components[qcomp_id].qgeometry_bounds()
            min_x_main = min(min_x, min_x_main)
            min_y_main = min(min_y, min_y_main)
            max_x_main = max(max_x, max_x_main)
            max_y_main = max(max_y, max_y_main)
    return min_x_main, min_y_main, max_x_main, max_y_main


def get_combined_dimensions(*components, unit: Optional[str] = None) -> dict:
    """Calculates combined dimensions and bounding box for a set of components.

    This wraps `get_min_bounding_box`.

    Args:
        *components: QComponent instances or sequences of them to be combined.

            Only components associated with a design and having an ID will be
            considered.
            A warning will be printed if any invalid one is found in the set.

        unit (str, optional): If specified (`"nm"`, `"um"`, `"mm"` et cætera), lengths
            are returned as formatted strings of `"{value} {unit}"`. Otherwise, returns
            raw float values in millimetres.

    Returns:
        dict: A dictionary containing:

            - 'width' (float or str): Combined width (x-axis span).
            - 'height' (float or str): Combined height (y-axis span).
            - 'bounds' (tuple): Bounding box coordinates,
              (min_x, min_y, max_x, max_y).

            If no valid components are passed, each of the quantities in the dictionary
            will be set to zero.

    """
    if not components:
        return {"width": 0.0, "height": 0.0, "bounds": (0.0, 0.0, 0.0, 0.0)}

    # Flatten nested lists/tuples of components.
    flat_components = []
    for c in components:
        if isinstance(c, (list, tuple)):
            flat_components.extend(c)
        else:
            flat_components.append(c)

    # Filter for components that actually have a registered design and ID.
    valid_components = []
    for c in flat_components:
        if hasattr(c, "design") and hasattr(c, "id"):
            valid_components.append(c)
        else:
            print(f"Invalid component: {c}")

    if not valid_components:
        return {"width": 0.0, "height": 0.0, "bounds": (0.0, 0.0, 0.0, 0.0)}

    # Get the parent design.
    design = valid_components[0].design
    comp_ids = [c.id for c in valid_components]

    # Find the axis-aligned bounding box using `get_min_bounding_box`.
    min_x, min_y, max_x, max_y = get_min_bounding_box(
        design, comp_ids, 0, design.logger
    )

    width = max_x - min_x
    height = max_y - min_y

    if unit is not None:
        unit_lower = unit.lower().strip()
        if unit_lower == "nm":
            factor = 1e6
            suffix = "nm"
        elif unit_lower == "um":
            factor = 1e3
            suffix = "um"
        elif unit_lower == "mm":
            factor = 1
            suffix = "mm"
        elif unit_lower == "m":
            factor = 1e-3
            suffix = "m"
        else:
            raise ValueError(
                f"Invalid unit: {unit}. Supported units are 'nm', 'um', 'mm', 'm'."
            )

        def fmt(val):
            return f"{val * factor:.{6}e} {suffix}"

        return {
            "width": fmt(width),
            "height": fmt(height),
            "bounds": (fmt(min_x), fmt(min_y), fmt(max_x), fmt(max_y)),
        }

    return {"width": width, "height": height, "bounds": (min_x, min_y, max_x, max_y)}
