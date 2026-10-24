"""Opt-in pure-Python implementations.

Import a concrete submodule by name. This package does not re-export any
symbols, so default ``refloxide`` / ``refloxide.tmm`` / ``refloxide.model``
callers cannot accidentally pick up the Python path.

Examples
--------
Pure-Python transfer-matrix kernel::

    from refloxide.python.tmm import uniaxial_reflectivity

Pure-Python modeling (pyref.fitting-shaped)::

    import refloxide.python.model as py

    vac = py.MaterialSLD("", density=0.0, energy=250.0, name="vacuum")
    model = py.ReflectModel(vac(0, 0) | ..., energy=250.0, pol="sp")
"""
