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
    wavelengths: Optional[np.ndarray],
    spectrum_columns: Optional[List[str]],
    spectrum_column: Optional[str],
    wavelength_column: Optional[str],
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

    if spectrum_column is not None:
        spectrum = np.asarray(row[spectrum_column], dtype=float)
    else:
        spectrum = np.array([row[c] for c in spectrum_columns], dtype=float)

    row_wavelengths = (
        np.asarray(row[wavelength_column], dtype=float)
        if wavelength_column is not None else wavelengths
    )

    pixel_id = row[pixel_id_column] if pixel_id_column is not None else row_idx
    metadata = {col: row[col] for col in metadata_columns}

    return Pixel(
        wavelengths=row_wavelengths,
        spectrum=spectrum,
        coordinates=geom,
        wavelength_unit=wavelength_unit,
        spectral_unit=spectral_unit,
        pixel_id=pixel_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# GeoDataFrame export (inverse of Dataset.from_geodataframe)
# ---------------------------------------------------------------------------


def pixels_to_geodataframe(pixels: List[Pixel], **extra_columns):
    """
    Convert a list of :class:`~specdec.Pixel` objects into a
    :mod:`geopandas` GeoDataFrame, one row per pixel with a non-``None``
    polygon (pixels with no geometry are silently skipped).

    This is the inverse of :meth:`Dataset.from_geodataframe`, and is the
    basis for the polygon-footprint plotting in :mod:`specdec.plotting`
    (e.g. :func:`~specdec.plotting.plot_abundance_map`) -- geopandas
    already correctly and efficiently renders many-pixel collections of
    arbitrary (possibly overlapping, possibly antimeridian-split
    ``MultiPolygon``) footprints as a single vectorised
    ``PatchCollection``, so there's no need for specdec to hand-roll its
    own per-pixel plotting loop.

    Parameters
    ----------
    pixels : list of Pixel
        Typically ``dataset.pixels`` or ``results.pixels``.
    **extra_columns
        Additional ``column_name=values`` to attach, where *values* is an
        array-like the same length as *pixels* (e.g. per-pixel abundance,
        RMS, or any other value to plot/inspect). Values are indexed down
        to the pixels that were actually kept (see above) automatically.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``pixel_id``, ``geometry`` (West-positive lon/lat, same
        convention as :attr:`Pixel.polygon`), plus any *extra_columns*.

    Raises
    ------
    ImportError
        If :mod:`geopandas` is not installed.
    ValueError
        If no pixel has a valid polygon, or an *extra_columns* array
        doesn't have one entry per pixel.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for pixels_to_geodataframe. "
            "Install it with: pip install specdec[geo]"
        )

    for name, values in extra_columns.items():
        if len(values) != len(pixels):
            raise ValueError(
                f"extra_columns[{name!r}] has length {len(values)}, "
                f"expected {len(pixels)} (one per pixel)."
            )

    keep_idx = [i for i, p in enumerate(pixels) if p.polygon is not None]
    if not keep_idx:
        raise ValueError("No pixels with valid geometry to build a GeoDataFrame from.")

    data = {
        "pixel_id": [pixels[i].pixel_id for i in keep_idx],
    }
    for name, values in extra_columns.items():
        values = np.asarray(values)
        data[name] = values[keep_idx]

    geometry = [pixels[i].polygon for i in keep_idx]
    return gpd.GeoDataFrame(data, geometry=geometry)


def resolve_overlaps(
    gdf,
    value_columns: List[str],
    weighting: str = "distance_x_inverse_area",
    coverage_buffer: float = 1e-3,
):
    """
    Decompose a GeoDataFrame of possibly-overlapping polygons into the
    arrangement of atomic (non-overlapping) regions implied by all of
    their boundaries together, with each atomic region's value(s) set to
    a weighted average of every original polygon's value that covers it.

    Rendering overlapping polygons directly (e.g. plain
    :func:`~specdec.plotting.plot_abundance_map`) always shows only
    whichever pixel happens to be drawn last/on top at a given point --
    everywhere pixels overlap, every other pixel's contribution there is
    completely hidden. This instead computes where the overlaps actually
    are and blends them, at the cost of splitting the input into many more
    (smaller, irregularly-shaped) pieces -- for a real HST long-slit
    dataset with heavy pixel overlap, expect roughly an order of magnitude
    more output rows than input rows, though this varies a lot with how
    much pixels actually overlap.

    Why a plain mean (or a single constant weight per polygon) still shows
    seams
    -----------------------------------------------------------------------
    Every atomic region blends its own, independent subset of covering
    polygons -- and in a dense, heavily-overlapping dataset (e.g. a
    pushbroom HST slit scan), that subset changes at *every* pixel edge,
    of which there are many, closely spaced. No single weight assigned per
    polygon (its full area, an emission-angle-based quality score,
    anything constant across its whole footprint) can smooth that out,
    because the discontinuity comes from *which* polygons are being
    averaged changing abruptly at each edge, not from how they're weighted
    relative to each other -- verified visually on a real dataset: a dense
    "houndstooth" lattice of hard edges tracing every pixel boundary,
    identical in character whether weighted by inverse area or left
    unweighted. Reducing that requires a weight that varies *within* a
    polygon's own footprint -- see ``weighting="distance"`` below.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygon geometries (Polygon or MultiPolygon) plus the columns named
        in *value_columns*. Any other columns are dropped.
    value_columns : list of str
        Numeric columns to average over each atomic region's covering
        polygons.
    weighting : str
        How to weight each covering polygon's contribution to an atomic
        region's average. One of:

        * ``"distance_x_inverse_area"`` (default) -- combines both terms
          below (multiplied together): tapers a polygon's influence
          smoothly to zero at its own boundary (killing hard seams) while
          still favouring smaller, less-foreshortened polygons overall
          when two polygons' interiors genuinely overlap.
        * ``"distance"`` -- weight by distance from the atomic region to
          the covering polygon's own boundary, normalised by
          :math:`\\sqrt{\\text{polygon area}}` so it's comparable in scale
          across differently-sized polygons: ~1 deep in a polygon's
          interior, ramping smoothly down to 0 at its edge. This alone is
          what actually removes seam artifacts -- a hard within/without
          cutoff becomes a smooth taper, so neighbouring atomic regions'
          blends no longer jump abruptly as a polygon's coverage begins or
          ends. Standard "distance feathering", as used for seamless
          orthophoto/image mosaicking.
        * ``"inverse_area"`` -- weight by the inverse of each covering
          polygon's own *full* (pre-overlap) footprint area (not the tiny
          atomic piece's area): a large, heavily-foreshortened polygon
          (e.g. a pixel near a disk's limb) represents one value smeared
          over a much bigger, less precise area than a small polygon near
          disk-center, so it's weighted down rather than counted equally.
          This alone does *not* remove seams (see above) -- it only
          changes which polygon dominates within a single atomic region's
          blend, not the discontinuity between neighbouring regions.
        * ``"none"`` -- plain unweighted mean.

    coverage_buffer : float
        Distance (in the units of *gdf*'s CRS -- degrees, for West-positive
        lon/lat data) that each input polygon is grown by *only* for the
        purpose of deciding which polygons cover a given atomic region.
        Adjacent input polygons that are meant to share an edge (e.g.
        consecutive along-slit pixel footprints) are frequently computed
        independently and so don't share bit-identical corner coordinates
        -- this leaves genuine, if tiny (sub-pixel-scale), gaps between
        them. Any atomic region that falls in one of those gaps has no
        *exact* covering polygon and would otherwise be dropped entirely,
        which is disproportionately noticeable: the gap regions tend to be
        thin, elongated slivers that trace right along the seam between
        polygons rather than small compact holes, so they render as
        conspicuous cracks through the middle of otherwise-solid coverage.
        A tiny buffer closes these without materially changing which
        *real*, non-adjacent polygons an atomic region matches, since
        genuinely distinct polygons are typically separated by orders of
        magnitude more than this. Verified across a real 664-pixel,
        heavily-overlapping HST dataset split into 4 per-visit subsets:
        this default closes every sliver gap with zero effect on an
        exact-grid (non-overlapping) 43,552-pixel dataset. Set to ``0`` to
        disable and require exact containment. Distances for
        ``weighting="distance"``/``"distance_x_inverse_area"`` are always
        measured against each polygon's true (unbuffered) boundary,
        regardless of this setting -- the buffer only affects which
        polygons are considered to cover a region at all.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per atomic region with coverage from at least one input
        polygon (regions outside every input polygon, even after
        *coverage_buffer* is applied, are not included), columns
        *value_columns* (each an average over covering polygons) and
        ``geometry``. Not guaranteed to preserve the input's row order or
        index.

    Raises
    ------
    ImportError
        If :mod:`geopandas` is not installed.
    ValueError
        If *value_columns* is empty, *weighting* is not a recognised
        option, or the arrangement has no atomic regions (e.g. *gdf* is
        empty).
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for resolve_overlaps. "
            "Install it with: pip install specdec[geo]"
        )
    import pandas as pd
    import shapely
    from shapely.ops import polygonize, unary_union

    if not value_columns:
        raise ValueError("value_columns must be a non-empty list of column names.")

    _valid_weightings = {"distance_x_inverse_area", "distance", "inverse_area", "none"}
    if weighting not in _valid_weightings:
        raise ValueError(
            f"weighting must be one of {sorted(_valid_weightings)}, got {weighting!r}."
        )

    geoms = list(gdf.geometry)
    # Snap each polygon's boundary to a fine coordinate grid individually,
    # *before* unioning them together. Without this, floating-point noise
    # can leave vertices that *should* coincide (e.g. two pixel corners
    # meant to touch) a few ulps apart instead, and polygonize() then
    # treats that as a real, separate feature: a near-zero-area sliver, or
    # even a same-ring "spike" that doubles back on itself for a few
    # vertices -- both numerically degenerate, so their winding direction
    # (exterior.is_ccw) is essentially noise, and a wrongly-signed ring is
    # exactly what makes cartopy's antimeridian-aware polygon fill (see
    # plotting._wpos_orient) invert one region into "fill the entire globe
    # except this shape". Snapping first collapses those near-duplicate
    # vertices so polygonize() never creates the degenerate region at all.
    # Precision-snapping the *already-unioned* multi-way boundary instead
    # (rather than each simple polygon boundary beforehand) looks
    # equivalent but isn't: reducing precision on a complex, already-noded
    # arrangement can corrupt its topology outright rather than just
    # tidying it -- verified on a real 3-pixel overlap where it collapsed
    # 14 correctly-reconstructed atomic faces (including the one actually
    # covering a real query point) into a single, wrong, unrelated blob.
    # Snapping each already-simple polygon boundary individually first
    # avoids that: 1e-9 degrees (~0.1 mm at Europa's surface) is far below
    # any real pixel-corner precision, so genuine geometry is untouched --
    # verified this leaves an exact-grid (non-overlapping) 43,552-pixel
    # dataset's atomic-region count unchanged, eliminates every degenerate
    # region found in a heavily-overlapping 145-pixel case, and (unlike
    # snapping post-union) recovers full coverage on a heavily-overlapping
    # 141-pixel case that includes 35 MultiPolygon (antimeridian-split)
    # footprints.
    snapped_boundaries = [shapely.set_precision(g.boundary, grid_size=1e-9) for g in geoms]
    boundaries = unary_union(snapped_boundaries)
    atomic = list(polygonize(boundaries))
    if not atomic:
        raise ValueError("No atomic regions found -- is gdf empty?")

    atomic_gdf = gpd.GeoDataFrame({"_atomic_id": range(len(atomic))}, geometry=atomic)

    # representative_point() (not centroid) is guaranteed to fall inside
    # the polygon even when it's non-convex.
    points_gdf = gpd.GeoDataFrame(
        {"_atomic_id": atomic_gdf["_atomic_id"]},
        geometry=atomic_gdf.geometry.representative_point(),
    )

    needs_area = weighting in ("inverse_area", "distance_x_inverse_area")
    needs_distance = weighting in ("distance", "distance_x_inverse_area")

    source = gdf[["geometry", *value_columns]].copy()
    if needs_area:
        source["_orig_area"] = gdf.geometry.area

    if coverage_buffer:
        coverage_gdf = gpd.GeoDataFrame(geometry=gdf.geometry.buffer(coverage_buffer))
        joined = gpd.sjoin(points_gdf, coverage_gdf, predicate="within", how="inner")
        joined = joined.join(source.drop(columns="geometry"), on="index_right")
    else:
        joined = gpd.sjoin(points_gdf, source, predicate="within", how="inner")

    if weighting == "none":
        averaged = joined.groupby("_atomic_id")[value_columns].mean()
    else:
        weight = np.ones(len(joined), dtype=float)

        if needs_distance:
            # Distance from each atomic region's representative point to
            # the *original* covering polygon's own boundary (never the
            # coverage_buffer-grown one -- that would shift the taper
            # outward/inward and defeat the point of measuring against the
            # polygon's true edge). Normalised by sqrt(area) so a small and
            # a large polygon's interiors both reach a comparable ~1 scale,
            # rather than the raw taper being dominated by whichever
            # polygon happens to be geometrically larger.
            covering_idx = joined["index_right"].to_numpy()
            covering_geoms = gdf.geometry.to_numpy()[covering_idx]
            covering_boundaries = shapely.boundary(covering_geoms)
            # joined's active geometry is inherited from points_gdf (the
            # left side of the sjoin) -- already the representative points,
            # aligned row-for-row with joined itself.
            pts = joined.geometry.to_numpy()
            dist = shapely.distance(covering_boundaries, pts)
            scale = np.sqrt(gdf.geometry.area.to_numpy()[covering_idx])
            # Guard against a zero-area sliver polygon making scale 0.
            scale = np.where(scale > 0, scale, 1.0)
            weight = weight * (dist / scale)

        if needs_area:
            weight = weight * (1.0 / joined["_orig_area"].to_numpy())

        # A representative point can legitimately land exactly on a
        # covering polygon's boundary (distance 0), zeroing that
        # contribution out entirely under "distance"/
        # "distance_x_inverse_area" -- fine as long as *some* covering
        # polygon has nonzero weight for that atomic region. Only guard
        # against the degenerate case where literally every contribution
        # to a region is zero, which would otherwise divide 0/0 to NaN.
        weight = np.where(weight > 0, weight, 1e-12)

        weight = pd.Series(weight, index=joined.index)
        weighted_sum = joined[value_columns].multiply(weight, axis=0).groupby(joined["_atomic_id"]).sum()
        weight_sum = weight.groupby(joined["_atomic_id"]).sum()
        averaged = weighted_sum.div(weight_sum, axis=0)

    result = atomic_gdf.merge(averaged, on="_atomic_id", how="inner")
    return result.drop(columns="_atomic_id")


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
        spectrum_columns: Optional[List[str]] = None,
        spectrum_column: Optional[str] = None,
        wavelengths=None,
        wavelength_column: Optional[str] = None,
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

        Spectra can be given in either of two common layouts -- pass
        exactly one of *spectrum_columns* or *spectrum_column*:

        - **Wide**: one column per wavelength band (``spectrum_columns``),
          e.g. ``['band_450', 'band_500', ...]``.
        - **Long/array**: a single column (``spectrum_column``) holding the
          whole spectrum as a 1-D array per row -- the natural layout when a
          GeoDataFrame was built alongside per-pixel geometry (e.g. via
          :mod:`planetspec`-style pipelines). Pair this with either a fixed
          *wavelengths* array (shared by every row) or a per-row
          *wavelength_column* if rows don't all share one grid.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input table.  The active geometry column must contain
            :class:`~shapely.geometry.Polygon` or
            :class:`~shapely.geometry.MultiPolygon` objects in the
            **positive-degrees-West** longitude convention (0 W → 360 W).
            Rows with null or unsupported geometry are skipped with a warning.
        spectrum_columns : list of str, optional
            Wide format: column names containing the per-band spectral
            values (one column per wavelength band, in wavelength order).
            Mutually exclusive with *spectrum_column*.
        spectrum_column : str, optional
            Long/array format: a single column whose values are a 1-D array
            of spectral values per row. Mutually exclusive with
            *spectrum_columns*.
        wavelengths : array-like, optional
            1-D array of wavelength values. With *spectrum_columns*: same
            length as *spectrum_columns*; if ``None``, parsed directly from
            the column names (which must be numeric strings, e.g.
            ``['450.0', '500.0', ...]``). With *spectrum_column*: a fixed
            grid shared by every row -- provide this or *wavelength_column*.
        wavelength_column : str, optional
            Only relevant with *spectrum_column*: a column whose values are
            a 1-D wavelength array per row, for datasets where rows don't
            all share the same grid. Takes precedence over *wavelengths*
            when both are given.
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
            If not exactly one of *spectrum_columns*/*spectrum_column* is
            given, if *wavelengths* length does not match
            *spectrum_columns*, if *wavelengths* cannot be parsed from the
            column names, or if *spectrum_column* is given without
            *wavelengths* or *wavelength_column*.

        Examples
        --------
        Wide format::

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

        Long/array format, one shared wavelength grid::

            ds = Dataset.from_geodataframe(
                gdf,
                spectrum_column="spectra",
                wavelengths=gdf["wavelength"].iloc[0],
                obs_id_columns=["visit"],
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

        # --- Resolve spectrum format (wide vs. long/array) --------------------
        have_wide = spectrum_columns is not None
        have_long = spectrum_column is not None
        if have_wide == have_long:
            raise ValueError(
                "Pass exactly one of spectrum_columns (wide: one column per "
                "band) or spectrum_column (long: a single array-valued "
                "column), not both or neither."
            )

        if have_wide:
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
        else:
            if wavelength_column is None and wavelengths is None:
                raise ValueError(
                    "spectrum_column requires either wavelengths (a fixed "
                    "array shared by every row) or wavelength_column (a "
                    "per-row array column)."
                )
            wavelengths = np.asarray(wavelengths, dtype=float) if wavelengths is not None else None

        metadata_columns = list(metadata_columns or [])

        def _make_pixel(row, row_idx):
            return _row_to_pixel(
                row, row_idx, wavelengths, spectrum_columns, spectrum_column,
                wavelength_column, geometry_column, pixel_id_column,
                wavelength_unit, spectral_unit, metadata_columns,
            )

        # --- Group rows into observations ------------------------------------
        dataset = cls()

        if obs_id_columns is None:
            obs = Observation()
            for row_idx, row in gdf.iterrows():
                pixel = _make_pixel(row, row_idx)
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
                    pixel = _make_pixel(row, row_idx)
                    if pixel is not None:
                        obs.add_pixel(pixel)
                dataset.add_observation(obs)

        return dataset

    # ------------------------------------------------------------------
    # GeoDataFrame export
    # ------------------------------------------------------------------

    def to_geodataframe(self, **extra_columns):
        """
        Convenience wrapper around :func:`pixels_to_geodataframe` using
        :attr:`pixels` (all pixels across all observations).

        See :func:`pixels_to_geodataframe` for parameters and behaviour.
        """
        return pixels_to_geodataframe(self.pixels, **extra_columns)

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
