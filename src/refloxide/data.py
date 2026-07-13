"""In-memory data containers: cached optical-constant tables and measured datasets.

`OpticalConstants` is the shared, polars-backed table wrapper every
dispersive `Scatterer` (built-in or user-defined) should hold instead of
loading and interpolating its own copy — see the caching contract on the
class itself.

`ReflectDataset` is the measured-data container `refloxide.objective.Objective`
consumes: it carries its own per-row polarization labeling, so which
channel(s) get fit is inferred from the dataset, never a constructor
argument on the model or objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

from refloxide import optics

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

Pol = Literal["s", "p"]


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
        # `table` is immutable for the lifetime of this (cached, shared)
        # instance, so converting each column to numpy once here -- instead
        # of on every `lookup`/`molecular_index_at` call -- avoids re-paying
        # polars' Series-to-numpy conversion on every single energy query.
        self._energy = table["energy"].to_numpy()
        self._n_xx = table["n_xx"].to_numpy()
        self._n_ixx = table["n_ixx"].to_numpy()
        self._n_zz = table["n_zz"].to_numpy()
        self._n_izz = table["n_izz"].to_numpy()

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
            self._energy, self._n_xx, self._n_ixx, self._n_zz, self._n_izz, energy_ev
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
            self._energy,
            self._n_xx,
            self._n_ixx,
            self._n_zz,
            self._n_izz,
            energy_ev,
            density,
        )


class ReflectDataset:
    """Measured reflectivity: q, energy, polarization channel, r, r_err.

    Rows are flat and independent — `energy` may repeat (many q points at
    one energy) and different energies may carry different numbers of
    points (ragged); there is no assumption of a rectangular `(q, energy)`
    grid. Each row's `pol` says which channel (`"s"` or `"p"`) it measures,
    which is how `refloxide.objective.Objective` infers what to evaluate
    without a `pol=` argument anywhere.

    Parameters
    ----------
    q, energy, r, r_err : array-like of float
        Flat, equal-length arrays.
    pol : array-like of {"s", "p"}
        Same length as `q`; the channel each row measures.

    Raises
    ------
    ValueError
        If the arrays are not all the same length, or `pol` holds a value
        other than `"s"`/`"p"`.
    """

    def __init__(
        self,
        q: NDArray[np.float64],
        energy: NDArray[np.float64],
        pol: NDArray[np.str_],
        r: NDArray[np.float64],
        r_err: NDArray[np.float64],
    ) -> None:
        self.q = np.asarray(q, dtype=np.float64).ravel()
        self.energy = np.asarray(energy, dtype=np.float64).ravel()
        self.pol = np.asarray(pol, dtype=object).ravel()
        self.r = np.asarray(r, dtype=np.float64).ravel()
        self.r_err = np.asarray(r_err, dtype=np.float64).ravel()

        lengths = {
            len(self.q),
            len(self.energy),
            len(self.pol),
            len(self.r),
            len(self.r_err),
        }
        if len(lengths) != 1:
            msg = (
                "ReflectDataset arrays must be the same length: "
                f"q={len(self.q)}, energy={len(self.energy)}, pol={len(self.pol)}, "
                f"r={len(self.r)}, r_err={len(self.r_err)}"
            )
            raise ValueError(msg)

        invalid = set(self.pol) - {"s", "p"}
        if invalid:
            msg = f"ReflectDataset pol must be 's' or 'p', got: {sorted(invalid)}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.q)

    @classmethod
    def from_polars(cls, frame: pl.DataFrame) -> ReflectDataset:
        """Build from a polars DataFrame with `q`/`energy`/`pol`/`r`/`r_err` columns.

        Parameters
        ----------
        frame : polars.DataFrame

        Returns
        -------
        ReflectDataset
        """
        required = {"q", "energy", "pol", "r", "r_err"}
        missing = required - set(frame.columns)
        if missing:
            msg = f"ReflectDataset frame missing columns: {sorted(missing)}"
            raise ValueError(msg)
        return cls(
            q=frame["q"].to_numpy(),
            energy=frame["energy"].to_numpy(),
            pol=frame["pol"].to_numpy(),
            r=frame["r"].to_numpy(),
            r_err=frame["r_err"].to_numpy(),
        )

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> ReflectDataset:
        """Build from a pandas DataFrame with `q`/`energy`/`pol`/`r`/`r_err` columns."""
        return cls.from_polars(pl.from_pandas(frame))

    @classmethod
    def from_arrays(
        cls,
        q: NDArray[np.float64],
        r: NDArray[np.float64],
        r_err: NDArray[np.float64],
        *,
        energy: float,
        pol: Pol,
    ) -> ReflectDataset:
        """Build a single-energy, single-channel dataset from plain arrays.

        Parameters
        ----------
        q, r, r_err : array-like of float
            Same length.
        energy : float
            Photon energy in eV, applied to every row.
        pol : {"s", "p"}
            Polarization channel, applied to every row.

        Returns
        -------
        ReflectDataset
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        return cls(
            q=q_arr,
            energy=np.full(q_arr.shape, float(energy)),
            pol=np.full(q_arr.shape, pol, dtype=object),
            r=r,
            r_err=r_err,
        )

    def groups(self) -> Iterator[tuple[float, Pol, NDArray[np.intp]]]:
        """Yield `(energy, pol, row_indices)` for each distinct `(energy, pol)` pair.

        Returns
        -------
        Iterator[tuple[float, str, NDArray[np.intp]]]
            One entry per unique `(energy, pol)` combination present in the
            dataset, with `row_indices` selecting that group's rows out of
            `q`/`r`/`r_err` in their original order.
        """
        keys = list(zip(self.energy.tolist(), self.pol.tolist(), strict=True))
        seen: dict[tuple[float, str], list[int]] = {}
        for i, key in enumerate(keys):
            seen.setdefault(key, []).append(i)
        for (energy, pol), indices in seen.items():
            yield energy, cast("Pol", pol), np.asarray(indices, dtype=np.intp)
