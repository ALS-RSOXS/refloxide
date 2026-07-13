"""In-memory data containers: cached optical-constant tables and measured datasets.

`OpticalConstants` is the shared, polars-backed table wrapper every
dispersive `Scatterer` (built-in or user-defined) should hold instead of
loading and interpolating its own copy — see the caching contract on the
class itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import polars as pl

from refloxide import optics


class OpticalConstants:
    """A cached, polars-backed table of tabulated optical constants for one material.

    This is a load-bearing guarantee, not an incidental optimization: every
    `from_file`/`from_dataframe`/`from_source` call that resolves to the
    same underlying material returns the SAME instance — not an equal one,
    the identical object — so N scatterers referencing the same material
    share one loaded, interpolatable table instead of each re-reading and
    re-interpolating it.

    Parameters
    ----------
    table : polars.DataFrame
        Table with columns `energy` (eV), `n_xx`, `n_ixx`, `n_zz`, `n_izz`
        — delta/beta for the ordinary (`xx`) and extraordinary (`zz`) axes.
    source : str
        Identity key used for caching — a resolved absolute file path, or
        an `id()`-derived key for an in-memory DataFrame passed directly.

    Raises
    ------
    ValueError
        If `table` is missing any of the required columns.
    """

    _cache: ClassVar[dict[str, OpticalConstants]] = {}
    _REQUIRED_COLUMNS = ("energy", "n_xx", "n_ixx", "n_zz", "n_izz")

    def __init__(self, table: pl.DataFrame, source: str):
        missing = set(self._REQUIRED_COLUMNS) - set(table.columns)
        if missing:
            columns = sorted(missing)
            raise ValueError(f"optical constants table missing columns: {columns}")
        self.table = table
        self.source = source

    @classmethod
    def from_file(cls, path: str | Path) -> OpticalConstants:
        """Load (or reuse) the table at `path`, keyed by its resolved absolute path.

        Parameters
        ----------
        path : str or pathlib.Path
            CSV path. `"znpc.csv"` and `"./znpc.csv"` called from the same
            working directory hit the same cache entry — the key is the
            path after `Path.resolve()`, not the raw string, so spelling
            differences can't accidentally defeat the cache and load two
            copies of one material.

        Returns
        -------
        OpticalConstants
        """
        key = str(Path(path).resolve())
        if key not in cls._cache:
            cls._cache[key] = cls(pl.read_csv(key), source=key)
        return cls._cache[key]

    @classmethod
    def from_dataframe(cls, table: pl.DataFrame) -> OpticalConstants:
        """Wrap an in-memory DataFrame, keyed by its identity.

        Parameters
        ----------
        table : polars.DataFrame

        Returns
        -------
        OpticalConstants
            Calling this twice with the SAME DataFrame object returns the
            same instance. Two separately-built DataFrames that happen to
            hold equal data are **not** deduplicated — there is no cheap
            way to know two DataFrame objects represent "the same
            material" without a shared key. Prefer `from_file` (path-keyed)
            when sharing across scatterers matters; if you must build the
            table in-memory, build it once and pass that one object
            everywhere it's needed.
        """
        key = f"<dataframe id={id(table)}>"
        if key not in cls._cache:
            cls._cache[key] = cls(table, source=key)
        return cls._cache[key]

    @classmethod
    def from_source(
        cls, source: OpticalConstants | pl.DataFrame | pd.DataFrame | str | Path
    ) -> OpticalConstants:
        """Accept an existing `OpticalConstants`, a polars/pandas DataFrame, or a path.

        Parameters
        ----------
        source : OpticalConstants, polars.DataFrame, pandas.DataFrame, str, or Path
            A pandas DataFrame is converted to polars via `pl.from_pandas`
            before being cached — the identity key is the pandas object's
            `id()`, so passing the same pandas DataFrame twice still shares
            one converted, cached table (see `from_dataframe`'s caveat:
            two separately-built DataFrames with equal content are not
            deduplicated, converted or not).

        Returns
        -------
        OpticalConstants
        """
        if isinstance(source, OpticalConstants):
            return source
        if isinstance(source, pl.DataFrame):
            return cls.from_dataframe(source)
        if isinstance(source, pd.DataFrame):
            key = f"<dataframe id={id(source)}>"
            if key not in cls._cache:
                cls._cache[key] = cls(pl.from_pandas(source), source=key)
            return cls._cache[key]
        return cls.from_file(source)

    @classmethod
    def cache_size(cls) -> int:
        """Number of distinct materials currently cached.

        Returns
        -------
        int
            Public introspection for the sharing guarantee — used to prove
            "three scatterers, one loaded table" rather than just asserting it.
        """
        return len(cls._cache)

    def lookup(self, energy_ev: float) -> tuple[float, float, float, float]:
        """Interpolate raw `(delta_xx, beta_xx, delta_zz, beta_zz)` at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV. Out-of-range values clamp to the
            tabulated endpoints (see `refloxide.optics.interp_ooc_linear`).

        Returns
        -------
        tuple[float, float, float, float]
            `(delta_xx, beta_xx, delta_zz, beta_zz)`, not density-scaled —
            use `molecular_index_at` for the density-scaled molecular index
            a `uniaxial_lab_tensor` call expects.
        """
        return optics.interp_ooc_linear(
            self.table["energy"].to_numpy(),
            self.table["n_xx"].to_numpy(),
            self.table["n_ixx"].to_numpy(),
            self.table["n_zz"].to_numpy(),
            self.table["n_izz"].to_numpy(),
            energy_ev,
        )

    def molecular_index_at(
        self, energy_ev: float, density: float
    ) -> tuple[complex, complex]:
        """Interpolate and density-scale to molecular `(n_xx, n_zz)` at `energy_ev`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.
        density : float
            Mass density scaling factor (g/cm^3).

        Returns
        -------
        tuple[complex, complex]
            `(n_mol_xx, n_mol_zz)`, ready to pass to
            `refloxide.optics.uniaxial_lab_tensor`.
        """
        return optics.molecular_index_at_ooc(
            self.table["energy"].to_numpy(),
            self.table["n_xx"].to_numpy(),
            self.table["n_ixx"].to_numpy(),
            self.table["n_zz"].to_numpy(),
            self.table["n_izz"].to_numpy(),
            energy_ev,
            density,
        )
