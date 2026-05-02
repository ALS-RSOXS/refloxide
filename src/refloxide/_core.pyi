from typing import Any

def hello_from_rust() -> str: ...
def compute_amplitudes_py(
    stack_repr: Any, omega_rad_per_s: float, theta_rad: float
) -> Any: ...
def compute_field_py(
    stack_repr: Any,
    omega_rad_per_s: float,
    theta_rad: float,
    layer_index: int,
    z_nm: float,
    polarization: Any,
) -> Any: ...
