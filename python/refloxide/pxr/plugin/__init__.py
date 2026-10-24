"""Plugin library for incorporation into the refnx framework."""

from refloxide.pxr.plugin.batched_global import (
    AnisotropyBatchTerm,
    BatchedFitter,
    BatchedGlobalObjective,
    ReflectivityBatchTerm,
    evaluate_reflectivity_batch,
)
from refloxide.pxr.plugin.dispersive_model import (
    BOOKENDED_FILM_PARAM_NAMES,
    REFLECT_MODEL_INSTRUMENTATION,
    DispersiveReflectModel,
    DispersiveReflectObjective,
    InstrumentFieldQuery,
    InstrumentParameterView,
    safely_setp_param,
    select,
)

__all__ = [
    "BOOKENDED_FILM_PARAM_NAMES",
    "REFLECT_MODEL_INSTRUMENTATION",
    "AnisotropyBatchTerm",
    "BatchedFitter",
    "BatchedGlobalObjective",
    "DispersiveReflectModel",
    "DispersiveReflectObjective",
    "InstrumentFieldQuery",
    "InstrumentParameterView",
    "ReflectivityBatchTerm",
    "evaluate_reflectivity_batch",
    "safely_setp_param",
    "select",
]
