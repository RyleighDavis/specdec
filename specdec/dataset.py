"""
Observation and Dataset classes.

An :class:`Observation` groups pixels that share a common viewing geometry or
instrument setup (e.g. one STIS slit position, one IFU pointing).  A
:class:`Dataset` is an ordered collection of one or more
:class:`Observation` objects.
"""

import warnings
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union

from shapely.geometry import MultiPolygon, Polygon

from .pixel import Pixel

try:
    from astropy.io import fits as astropy_fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class Observation:
    """
    A single observation — an ordered collection of :class:`~specdec.Pixel`
    objects that share a common viewing geometry or instrument configuration.

    Examples of a single observation include one slit position of an HST/STIS
    long-slit spectrum, one IFU pointing with a fixed disk-centre and
    viewing geometry, or one raster row of a push-broom spectrometer.

    Parameters
    ----------
    pixels : list of Pixel, optional
        Initial pixels belonging to this observation.
    observation_info : dict, optional
        Metadata describing the observing geometry / configuration common to
        all pixels in this observation.  Typical keys:

        ``'telescope'``, ``'instrument'``, ``'target'``, ``'date_obs'``,
        ``'slit_id'``, ``'disk_center_lon'``, ``'disk_center_lat'``,
        ``'phase_angle'``, ``'sub_obs_lon'``, …
    obs_id : any, optional
        Arbitrary unique identifier for this observation (e.g. a slit number,
        an exposure ID, or a string label).
    """

    def __init__(
        self,
        pixels: Optional[List[Pixel]] = None,
        observation_info: Optional[Dict[str, Any]] = None,
        obs_id=None,
    ):
        self._pixels: List[Pixel] = []
        self.observation_info: Dict[str, Any] = observation_info or {}
        self.obs_id = obs_id

        if pixels is not None:
            for p in pixels:
                self.add_pixel(p)

    # ------------------------------------------------------------------
    # Pixel management
    # ------------------------------------------------------------------

    def add_pixel(self, pixel: Pixel):
        """Append a single :class:`~specdec.Pixel` to this observation."""
        if not isinstance(pixel, Pixel):
            raise TypeError(f"Expected a Pixel, got {type(pixel).__name__}.")
        self._pixels.append(pixel)

    def add_pixels(self, pixels):
        """Append multiple :class:`~specdec.Pixel` objects."""
        for p in pixels:
            self.add_pixel(p)

    # ------------------------------------------------------------------
    # Pixel views
    # ------------------------------------------------------------------

    @property
    def pixels(self) -> List[Pixel]:
        """All pixels in this observation, in insertion order."""
        return self._pixels

    @property
    def candidate_pixels(self) -> List[Pixel]:
        """Pixels currently eligible for endmember selection."""
        return [p for p in self._pixels if p.is_candidate]

    @property
    def excluded_pixels(self) -> List[Pixel]:
        """Pixels that have been excluded from endmember selection."""
        return [p for p in self._pixels if not p.is_candidate]

    # ------------------------------------------------------------------
    # FITS loader
    # ------------------------------------------------------------------

    @classmethod
    def from_fits(
        cls,
        filepath: str,
        observation_info: Optional[Dict[str, Any]] = None,
        obs_id=None,
        **kwargs,
    ) -> "Observation":
        """
        Create an :class:`Observation` whose ``observation_info`` is populated
        from the primary FITS header.

        .. note::
            This is a minimal loader — it attaches the FITS primary header as
            observation metadata but does **not** populate pixels automatically.
            For instrument-specific formats (HST/STIS, JWST NIRSpec IFU, etc.)
            subclass :class:`Observation` and override this method, or build
            pixels manually after opening the file with :mod:`astropy.io.fits`.

        Parameters
        ----------
        filepath : str
            Path to the FITS file.
        observation_info : dict, optional
            Additional metadata merged *on top of* the FITS header (takes
            precedence on key conflicts).
        obs_id : any, optional
            Identifier for this observation.

        Requires
        --------
        ``astropy`` — install with ``pip install specdec[fits]``.
        """
        if not HAS_ASTROPY:
            raise ImportError(
                "astropy is required for FITS loading. "
                "Install it with: pip install specdec[fits]"
            )
        with astropy_fits.open(filepath) as hdul:
            header_dict = dict(hdul[0].header)

        combined_info = dict(header_dict)
        if observation_info:
            combined_info.update(observation_info)

        return cls(observation_info=combined_info, obs_id=obs_id, **kwargs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pixels)

    def __iter__(self):
        return iter(self._pixels)

    def __getitem__(self, index):
        return self._pixels[index]

    def __repr__(self) -> str:
        return (
            f"Observation(obs_id={self.obs_id!r}, "
            f"n_pixels={len(self._pixels)}, "
            f"n_candidates={len(self.candidate_pixels)}, "
            f"n_excluded={len(self.excluded_pixels)})"
        )


# ---------------------------------------------------------------------------
# GeoDataFrame row helper (module-level for clarity)
# ---------------------------------------------------------------------------


def _row_to_pixel(
    row,
    row_idx,
    wavelengths: np.ndarray,
    spectrum_columns: List[str],
    geometry_column: str,
    pixel_id_column: Optional[str],
    wavelength_unit: str,
    spectral_unit: str,
    metadata_columns: List[str],
) -> Optional[Pixel]:
    """
    Convert a single GeoDataFrame row to a :class:`~specdec.Pixel`.

    Returns ``None`` (with a warning) if the geometry is null or not a
    Polygon / MultiPolygon.
    """
    geom = row[geometry_column]

    # Skip null geometries
    if geom is None or (hasattr(geom, "is_empty") and geom is None):
        warnings.warn(
            f"Row {row_idx!r} has a null geometry — skipped.",
            stacklevel=4,
        )
        return None

    try:
        import pandas as pd
        if pd.isna(geom):
            warnings.warn(
                f"Row {row_idx!r} has a null geometry — skipped.",
                stacklevel=4,
            )
            return None
    except (TypeError, ValueError):
        pass  # pd.isna doesn't work on geometry objects in newer geopandas

    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise TypeError(
            f"Row {row_idx!r}: expected Polygon or MultiPolygon geometry, "
            f"got {type(geom).__name__}. Convert point/line geometries to "
            "polygon footprints before loading."
        )

    spectrum = np.array([row[c] for c in spectrum_columns], dtype=float)
    pixel_id = row[pixel_id_column] if pixel_id_column is not None else row_idx
    metadata = {col: row[col] for col in metadata_columns}

    return Pixel(
        wavelengths=wavelengths,
        spectrum=spectrum,
        coordinates=geom,
        wavelength_unit=wavelength_unit,
        spectral_unit=spectral_unit,
        pixel_id=pixel_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset:
    """
    An ordered collection of :class:`Observation` objects.

    A :class:`Dataset` aggregates pixels from multiple observations that may
    differ in slit position, IFU pointing, disk-centre location, or any other
    per-observation geometry.  All pixel-level operations (exclusion, spectra
    matrices, etc.) span the full collection.

    Parameters
    ----------
    observations : list of Observation, optional
        Initial observations to add.

    Examples
    --------
    Build a dataset from two STIS slit positions::

        obs1 = Observation(observation_info={'slit_id': 1, 'disk_center_lon': 0.0})
        obs2 = Observation(observation_info={'slit_id': 2, 'disk_center_lon': 5.0})

        for wl, spec, coords, ea in slit1_data:
            obs1.add_pixel(Pixel(wl, spec, coordinates=coords,
                                 metadata={'emission_angle': ea}))

        for wl, spec, coords, ea in slit2_data:
            obs2.add_pixel(Pixel(wl, spec, coordinates=coords,
                                 metadata={'emission_angle': ea}))

        ds = Dataset([obs1, obs2])
        ds.exclude_by_metadata('emission_angle', '>', 60)

    Notes
    -----
    Pixel exclusions (candidate status) are stored on individual
    :class:`~specdec.Pixel` objects.  The snapshot of which pixels are
    candidates is taken when an :class:`~specdec.EndmemberDecomposition` is
    constructed; changes made *after* that point will not be reflected
    automatically — create a new :class:`~specdec.EndmemberDecomposition`
    instead.
    """

    def __init__(self, observations: Optional[List[Observation]] = None):
        self._observations: List[Observation] = []

        if observations is not None:
            for obs in observations:
                self.add_observation(obs)

    # ------------------------------------------------------------------
    # Observation management
    # ------------------------------------------------------------------

    def add_observation(self, observation: Observation):
        """Append a single :class:`Observation`."""
        if not isinstance(observation, Observation):
            raise TypeError(
                f"Expected an Observation, got {type(observation).__name__}."
            )
        self._observations.append(observation)

    def add_observations(self, observations):
        """Append multiple :class:`Observation` objects."""
        for obs in observations:
            self.add_observation(obs)

    # ------------------------------------------------------------------
    # Observation views
    # ------------------------------------------------------------------

    @property
    def observations(self) -> List[Observation]:
        """All observations, in insertion order."""
        return self._observations

    # ------------------------------------------------------------------
    # Flat pixel views (across all observations)
    # ------------------------------------------------------------------

    @property
    def pixels(self) -> List[Pixel]:
        """All pixels across all observations, in insertion order."""
        return [p for obs in self._observations for p in obs.pixels]

    @property
    def candidate_pixels(self) -> List[Pixel]:
        """Pixels currently eligible for endmember selection (all observations)."""
        return [p for obs in self._observations for p in obs.candidate_pixels]

    @property
    def excluded_pixels(self) -> List[Pixel]:
        """Excluded pixels across all observations."""
        return [p for obs in self._observations for p in obs.excluded_pixels]

    # ------------------------------------------------------------------
    # Convenience: add a pixel directly (auto-creates a bare observation)
    # ------------------------------------------------------------------

    def add_pixel(self, pixel: Pixel, obs_id=None):
        """
        Append a single :class:`~specdec.Pixel`, optionally routing it to an
        existing observation matched by *obs_id*.

        If *obs_id* is ``None`` **and** the dataset already contains exactly
        one observation, the pixel is appended to that observation.  Otherwise
        a new bare :class:`Observation` is created.

        Parameters
        ----------
        pixel : Pixel
        obs_id : any, optional
            Target observation identifier.  If no observation with this ID
            exists, a new :class:`Observation` is created with that ID.
        """
        if obs_id is None:
            if len(self._observations) == 1:
                self._observations[0].add_pixel(pixel)
                return
            elif len(self._observations) == 0:
                obs = Observation(obs_id=None)
                self._observations.append(obs)
                obs.add_pixel(pixel)
                return
            else:
                raise ValueError(
                    "Dataset has multiple observations; supply obs_id to route "
                    "the pixel to the correct one."
                )

        # Find or create observation with matching obs_id
        for obs in self._observations:
            if obs.obs_id == obs_id:
                obs.add_pixel(pixel)
                return

        new_obs = Observation(obs_id=obs_id)
        self._observations.append(new_obs)
        new_obs.add_pixel(pixel)

    # ------------------------------------------------------------------
    # Exclusion helpers (operate across all pixels in all observations)
    # ------------------------------------------------------------------

    def exclude_pixels(self, criterion: Callable[[Pixel], bool]) -> int:
        """
        Exclude pixels whose criterion callable returns ``True``.

        Parameters
        ----------
        criterion : callable
            ``criterion(pixel) -> bool``.  Return ``True`` to exclude.

        Returns
        -------
        int
            Number of pixels excluded by this call.

        Examples
        --------
        >>> ds.exclude_pixels(lambda p: p.metadata.get('emission_angle', 0) > 60)
        """
        count = 0
        for pixel in self.pixels:
            if criterion(pixel):
                pixel.exclude()
                count += 1
        return count

    def include_pixels(self, criterion: Callable[[Pixel], bool]) -> int:
        """
        Re-include pixels whose criterion callable returns ``True``.

        Parameters
        ----------
        criterion : callable
            ``criterion(pixel) -> bool``.  Return ``True`` to re-include.

        Returns
        -------
        int
            Number of pixels re-included by this call.
        """
        count = 0
        for pixel in self.pixels:
            if criterion(pixel):
                pixel.include()
                count += 1
        return count

    def exclude_by_metadata(self, key: str, operator: str, value: float) -> int:
        """
        Exclude pixels based on a numeric metadata field comparison.

        Parameters
        ----------
        key : str
            Metadata key to compare (e.g. ``'emission_angle'``).
        operator : str
            One of ``'>'``, ``'<'``, ``'>='``, ``'<='``, ``'=='``, ``'!='``.
        value : float
            Threshold value.

        Returns
        -------
        int
            Number of pixels excluded.

        Examples
        --------
        >>> ds.exclude_by_metadata('emission_angle', '>', 60)
        3
        """
        _ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        if operator not in _ops:
            raise ValueError(
                f"operator must be one of {list(_ops.keys())}, got {operator!r}."
            )
        op_fn = _ops[operator]

        def criterion(pixel: Pixel) -> bool:
            val = pixel.metadata.get(key)
            if val is None:
                return False
            return bool(op_fn(val, value))

        return self.exclude_pixels(criterion)

    def reset_exclusions(self):
        """Re-include all pixels across all observations."""
        for pixel in self.pixels:
            pixel.include()

    # ------------------------------------------------------------------
    # Spectra matrix helpers
    # ------------------------------------------------------------------

    def get_pixel(self, pixel_id) -> Pixel:
        """
        Return the first pixel whose ``pixel_id`` matches *pixel_id*.

        Parameters
        ----------
        pixel_id : any
            The identifier to search for.

        Returns
        -------
        Pixel

        Raises
        ------
        KeyError
            If no pixel with that ``pixel_id`` exists in the dataset.
        """
        for p in self.pixels:
            if p.pixel_id == pixel_id:
                return p
        raise KeyError(f"No pixel with pixel_id={pixel_id!r} found in dataset.")

    def get_spectra_matrix(
        self, pixels: Optional[List[Pixel]] = None
    ) -> np.ndarray:
        """
        Return spectra stacked into a 2-D array.

        Parameters
        ----------
        pixels : list of Pixel, optional
            Subset to include.  Defaults to :attr:`pixels` (all pixels across
            all observations).

        Returns
        -------
        ndarray, shape (n_pixels, n_wavelengths)
        """
        if pixels is None:
            pixels = self.pixels
        if not pixels:
            return np.empty((0, 0))
        return np.vstack([p.spectrum for p in pixels])

    # ------------------------------------------------------------------
    # GeoDataFrame loader
    # ------------------------------------------------------------------

    @classmethod
    def from_geodataframe(
        cls,
        gdf,
        spectrum_columns: List[str],
        wavelengths=None,
        obs_id_columns: Optional[List[str]] = None,
        pixel_id_column: Optional[str] = None,
        wavelength_unit: str = "nm",
        spectral_unit: str = "reflectance",
        metadata_columns: Optional[List[str]] = None,
        geometry_column: Optional[str] = None,
    ) -> "Dataset":
        """
        Build a :class:`Dataset` from a :mod:`geopandas` GeoDataFrame.

        Each row becomes one :class:`~specdec.Pixel`.  Rows are optionally
        grouped into :class:`Observation` objects by one or more columns;
        the unique combination of those column values defines a single
        observation.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input table.  The active geometry column must contain
            :class:`~shapely.geometry.Polygon` or
            :class:`~shapely.geometry.MultiPolygon` objects in the
            **positive-degrees-West** longitude convention (0 W → 360 W).
            Rows with null or unsupported geometry are skipped with a warning.
        spectrum_columns : list of str
            Column names containing the per-band spectral values (one column
            per wavelength band, in wavelength order).
        wavelengths : array-like, optional
            1-D array of wavelength values (same length as
            *spectrum_columns*).  If ``None``, the values are parsed
            directly from *spectrum_columns* — the column names must be
            numeric strings (e.g. ``['450.0', '500.0', ...]``).
        obs_id_columns : list of str, optional
            Column names whose unique value combinations define individual
            :class:`Observation` objects (e.g. ``['obs_id', 'slit_id']``).
            Rows sharing identical values for *all* of these columns are
            placed into the same observation, with those values stored in
            the observation's ``observation_info`` dict.  If ``None``, all
            pixels are placed in a single observation.
        pixel_id_column : str, optional
            Column to use for :attr:`~specdec.Pixel.pixel_id`.  Defaults to
            the DataFrame row index when ``None``.
        wavelength_unit : str
            Wavelength unit string applied to all pixels.  Default ``'nm'``.
        spectral_unit : str
            Spectral unit string applied to all pixels.
            Default ``'reflectance'``.
        metadata_columns : list of str, optional
            Extra columns to include in each pixel's
            :attr:`~specdec.Pixel.metadata` dict.
        geometry_column : str, optional
            Name of the geometry column.  Defaults to the GeoDataFrame's
            active geometry column (``gdf.geometry.name``).

        Returns
        -------
        Dataset

        Raises
        ------
        ImportError
            If :mod:`geopandas` is not installed.
        ValueError
            If *wavelengths* length does not match *spectrum_columns*, or if
            *wavelengths* cannot be parsed from the column names.

        Examples
        --------
        ::

            import geopandas as gpd
            from specdec import Dataset

            gdf = gpd.read_file("spectra.gpkg")
            band_cols = [c for c in gdf.columns if c.startswith("band_")]

            ds = Dataset.from_geodataframe(
                gdf,
                spectrum_columns=band_cols,
                wavelengths=np.linspace(400, 900, len(band_cols)),
                obs_id_columns=["obs_id", "slit_id"],
                pixel_id_column="pixel_id",
                metadata_columns=["emission_angle", "phase_angle"],
            )
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for Dataset.from_geodataframe. "
                "Install it with: pip install specdec[geo]"
            )

        # --- Resolve geometry column -----------------------------------------
        if geometry_column is None:
            geometry_column = gdf.geometry.name

        # --- Resolve wavelengths ---------------------------------------------
        if wavelengths is None:
            try:
                wavelengths = np.array([float(c) for c in spectrum_columns])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "Could not parse wavelengths from spectrum_columns names. "
                    "Provide wavelengths explicitly."
                ) from exc
        else:
            wavelengths = np.asarray(wavelengths, dtype=float)

        if len(wavelengths) != len(spectrum_columns):
            raise ValueError(
                f"len(wavelengths) ({len(wavelengths)}) must equal "
                f"len(spectrum_columns) ({len(spectrum_columns)})."
            )

        metadata_columns = list(metadata_columns or [])

        # --- Group rows into observations ------------------------------------
        dataset = cls()

        if obs_id_columns is None:
            obs = Observation()
            for row_idx, row in gdf.iterrows():
                pixel = _row_to_pixel(
                    row, row_idx, wavelengths, spectrum_columns,
                    geometry_column, pixel_id_column,
                    wavelength_unit, spectral_unit, metadata_columns,
                )
                if pixel is not None:
                    obs.add_pixel(pixel)
            dataset.add_observation(obs)
        else:
            for group_keys, group_df in gdf.groupby(obs_id_columns, sort=True):
                # Normalise to tuple for uniform handling
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)
                obs_info = dict(zip(obs_id_columns, group_keys))
                obs_id = "_".join(str(v) for v in group_keys)
                obs = Observation(observation_info=obs_info, obs_id=obs_id)
                for row_idx, row in group_df.iterrows():
                    pixel = _row_to_pixel(
                        row, row_idx, wavelengths, spectrum_columns,
                        geometry_column, pixel_id_column,
                        wavelength_unit, spectral_unit, metadata_columns,
                    )
                    if pixel is not None:
                        obs.add_pixel(pixel)
                dataset.add_observation(obs)

        return dataset

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of pixels across all observations."""
        return sum(len(obs) for obs in self._observations)

    def __iter__(self):
        """Iterate over :class:`Observation` objects."""
        return iter(self._observations)

    def __getitem__(self, index):
        """Index into the list of observations."""
        return self._observations[index]

    def __repr__(self) -> str:
        n_pixels = len(self)
        n_candidates = len(self.candidate_pixels)
        return (
            f"Dataset(n_observations={len(self._observations)}, "
            f"n_pixels={n_pixels}, "
            f"n_candidates={n_candidates}, "
            f"n_excluded={n_pixels - n_candidates})"
        )
