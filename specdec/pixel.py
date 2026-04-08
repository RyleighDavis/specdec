"""
Pixel class representing a single spatial pixel with spectral data.

Coordinate convention
---------------------
Longitudes are stored in **positive-degrees-West** (0 W – 360 W).
Latitudes are in degrees North (−90 – 90).

When corners cross the antimeridian (180 W) or the prime-meridian wrap-around
(0 W / 360 W), the stored ``polygon`` is automatically split into a
:class:`~shapely.geometry.MultiPolygon` so that each part can be rendered
correctly without spanning the full map.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import split as shapely_split, unary_union
from shapely.geometry import LineString


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------


def _detect_crossing(lons: List[float]) -> str:
    """
    Detect whether a set of corner longitudes crosses a map boundary.

    Returns
    -------
    'none'      No boundary crossing detected.
    '180W'      Polygon straddles the antimeridian (180 W).
    '0W360W'    Polygon straddles the prime-meridian wrap-around (0 W / 360 W).
    """
    lo, hi = min(lons), max(lons)
    span = hi - lo
    if span > 180:
        # Large span ⟹ polygon wraps around the 0 W / 360 W boundary.
        return "0W360W"
    if lo < 180.0 < hi:
        return "180W"
    return "none"


def _split_polygon_at_lon(polygon: Polygon, lon: float) -> List[Polygon]:
    """
    Split *polygon* (W-space) at a meridian of fixed longitude *lon*.

    Returns a list of 1 or 2 :class:`~shapely.geometry.Polygon` objects.
    Falls back to a single-element list if the split cannot be performed.
    """
    split_line = LineString([(lon, -91.0), (lon, 91.0)])
    try:
        result = shapely_split(polygon, split_line)
        polys = [g for g in result.geoms if isinstance(g, Polygon)]
        return polys if polys else [polygon]
    except Exception:
        return [polygon]


def _split_polygon_prime_meridian(polygon: Polygon) -> List[Polygon]:
    """
    Split *polygon* at the 0 W / 360 W boundary.

    Shifts all longitudes by +180 (mod 360) so the 0 W / 360 W wrap-around
    maps to the 180 W meridian, splits there, then shifts back.
    """
    # Shift forward so 0/360 → 180
    shifted_ext = [((lon + 180.0) % 360.0, lat) for lon, lat in polygon.exterior.coords]
    shifted_int = [
        [((lon + 180.0) % 360.0, lat) for lon, lat in ring.coords]
        for ring in polygon.interiors
    ]
    shifted_poly = Polygon(shifted_ext, shifted_int)

    pieces_shifted = _split_polygon_at_lon(shifted_poly, 180.0)

    # Shift back: lon_original = (lon_shifted − 180) mod 360
    result = []
    for piece in pieces_shifted:
        back_ext = [((lon - 180.0) % 360.0, lat) for lon, lat in piece.exterior.coords]
        back_int = [
            [((lon - 180.0) % 360.0, lat) for lon, lat in ring.coords]
            for ring in piece.interiors
        ]
        result.append(Polygon(back_ext, back_int))
    return result


def _build_polygon_from_corners(
    corners: List[Tuple[float, float]],
) -> Union[Polygon, MultiPolygon]:
    """
    Build a (possibly split) shapely geometry from W-space corner coordinates.

    If the polygon crosses the antimeridian (180 W) or the prime-meridian
    wrap-around (0 W / 360 W), it is split and a
    :class:`~shapely.geometry.MultiPolygon` is returned.
    """
    lons = [c[0] for c in corners]
    polygon = Polygon(corners)

    crossing = _detect_crossing(lons)

    if crossing == "180W":
        pieces = _split_polygon_at_lon(polygon, 180.0)
    elif crossing == "0W360W":
        pieces = _split_polygon_prime_meridian(polygon)
    else:
        return polygon

    if len(pieces) == 1:
        return pieces[0]
    return MultiPolygon(pieces)


def _centroid_from_corners(corners: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Compute the centroid of corner coordinates in W-space, correctly handling
    wrap-around at the 0 W / 360 W boundary via a circular mean.
    """
    lons = np.asarray([c[0] for c in corners])
    lats = np.asarray([c[1] for c in corners])
    lat_mean = float(np.mean(lats))

    if lons.max() - lons.min() > 180.0:
        # 0 W / 360 W crossing — use circular mean
        shifted = (lons + 180.0) % 360.0
        lon_mean = (float(np.mean(shifted)) - 180.0) % 360.0
    else:
        lon_mean = float(np.mean(lons))

    return (lon_mean, lat_mean)


# ---------------------------------------------------------------------------
# Pixel class
# ---------------------------------------------------------------------------


class Pixel:
    """
    A single spatial pixel with associated wavelength-dependent spectral data.

    Coordinates must be supplied as a list of ≥ 3 ``(longitude, latitude)``
    corner tuples (in positive-degrees-West / degrees-North convention) **or**
    as a pre-built :class:`~shapely.geometry.Polygon` /
    :class:`~shapely.geometry.MultiPolygon`.  Single centroid points are not
    accepted; if a pixel has only a known centroid, construct a small bounding
    polygon around it.

    Parameters
    ----------
    wavelengths : array-like
        1-D wavelength array.
    spectrum : array-like
        1-D spectral data array (same length as *wavelengths*).
    coordinates : list of (lon, lat) tuples | shapely Polygon | None
        Spatial footprint of the pixel.  Longitudes in positive-degrees-West
        (0 W → 360 W), latitudes in degrees North (−90 → 90).

        * A list of ≥ 3 ``(longitude, latitude)`` tuples is converted to a
          :class:`~shapely.geometry.Polygon` automatically.  If the polygon
          crosses the antimeridian (180 W) or the prime-meridian wrap-around
          (0 W / 360 W) it is split into a
          :class:`~shapely.geometry.MultiPolygon`.
        * A shapely :class:`~shapely.geometry.Polygon` or
          :class:`~shapely.geometry.MultiPolygon` is stored as-is (no
          automatic splitting is performed).
        * ``None`` — no spatial information.
    wavelength_unit : str
        Unit string for wavelengths (e.g. ``'nm'``, ``'micron'``).
        Default ``'nm'``.
    spectral_unit : str
        Unit string for the spectrum (e.g. ``'reflectance'``, ``'I/F'``).
        Default ``'reflectance'``.
    pixel_id : any, optional
        Arbitrary unique identifier.
    metadata : dict, optional
        Arbitrary key/value pairs (e.g. ``{'emission_angle': 35.2}``).
    is_candidate : bool
        Whether this pixel is eligible to be selected as an endmember.
        Default ``True``.
    """

    def __init__(
        self,
        wavelengths,
        spectrum,
        coordinates=None,
        wavelength_unit: str = "nm",
        spectral_unit: str = "reflectance",
        pixel_id=None,
        metadata: Optional[Dict[str, Any]] = None,
        is_candidate: bool = True,
    ):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.spectrum = np.asarray(spectrum, dtype=float)

        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1-D array.")
        if self.spectrum.ndim != 1:
            raise ValueError("spectrum must be a 1-D array.")
        if len(self.wavelengths) != len(self.spectrum):
            raise ValueError(
                f"wavelengths (len {len(self.wavelengths)}) and spectrum "
                f"(len {len(self.spectrum)}) must have the same length."
            )

        finite_mask = np.isfinite(self.spectrum)
        if not np.any(finite_mask):
            raise ValueError(
                "All spectral values are non-finite (NaN or Inf). "
                "Be sure to filter out all-NaN spectra before creating a Pixel object."
            )
        if not np.all(finite_mask):
            n_bad = int((~finite_mask).sum())
            warnings.warn(
                f"Pixel {pixel_id!r}: {n_bad} non-finite spectral value(s) "
                "interpolated from neighbouring wavelengths.",
                stacklevel=2,
            )
            self.spectrum = np.interp(
                self.wavelengths,
                self.wavelengths[finite_mask],
                self.spectrum[finite_mask],
            )

        self.wavelength_unit = wavelength_unit
        self.spectral_unit = spectral_unit
        self.pixel_id = pixel_id
        self.metadata: Dict[str, Any] = metadata or {}
        self._is_candidate: bool = bool(is_candidate)

        # Coordinate storage
        self._polygon: Optional[Union[Polygon, MultiPolygon]] = None
        self._corners: Optional[List[Tuple[float, float]]] = None
        self._centroid: Optional[Tuple[float, float]] = None
        self._set_coordinates(coordinates)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _set_coordinates(self, coordinates) -> None:
        if coordinates is None:
            return

        # Shapely polygon / multipolygon supplied directly
        if isinstance(coordinates, (Polygon, MultiPolygon)):
            self._polygon = coordinates
            # centroid computed on-demand from shapely
            return

        if isinstance(coordinates, (list, tuple)):
            # Reject bare (lon, lat) centroid points — no longer supported
            if (
                len(coordinates) == 2
                and not isinstance(coordinates[0], (list, tuple, np.ndarray))
            ):
                raise TypeError(
                    "Single (longitude, latitude) point coordinates are no longer "
                    "supported.  Provide corners as a list of ≥ 3 (lon, lat) pairs "
                    "or a shapely Polygon."
                )

            corners = []
            for item in coordinates:
                if not hasattr(item, "__len__") or len(item) != 2:
                    raise ValueError(
                        "Each corner must be a (longitude, latitude) pair."
                    )
                corners.append((float(item[0]), float(item[1])))

            if len(corners) < 3:
                raise ValueError(
                    "Corner list must have at least 3 points to define a polygon."
                )

            self._corners = corners
            self._centroid = _centroid_from_corners(corners)
            self._polygon = _build_polygon_from_corners(corners)
            return

        raise TypeError(
            "coordinates must be a list of (lon, lat) corner tuples or a shapely "
            f"Polygon / MultiPolygon.  Got {type(coordinates).__name__}."
        )

    # ------------------------------------------------------------------
    # Public coordinate properties
    # ------------------------------------------------------------------

    @property
    def polygon(self) -> Optional[Union[Polygon, MultiPolygon]]:
        """
        Shapely geometry for this pixel's spatial footprint, or ``None``.

        If corners were supplied that cross the antimeridian or the
        0 W / 360 W boundary, this is a
        :class:`~shapely.geometry.MultiPolygon`; otherwise a
        :class:`~shapely.geometry.Polygon`.
        """
        return self._polygon

    @property
    def corners(self) -> Optional[List[Tuple[float, float]]]:
        """
        Original ``(lon, lat)`` corner tuples as supplied by the user, or
        ``None`` if a :class:`~shapely.geometry.Polygon` was supplied directly.
        """
        return self._corners

    @property
    def centroid(self) -> Optional[Tuple[float, float]]:
        """
        ``(longitude, latitude)`` centroid in positive-degrees-West / degrees-North.

        * Computed from the original corner coordinates (with circular-mean
          longitude for pixels crossing the 0 W / 360 W boundary) when corners
          were supplied.
        * Computed from the shapely polygon's centroid when a
          :class:`~shapely.geometry.Polygon` was supplied directly.
        * ``None`` if no coordinates were given.
        """
        if self._centroid is not None:
            return self._centroid
        if self._polygon is not None:
            c = self._polygon.centroid
            return (c.x, c.y)
        return None

    # ------------------------------------------------------------------
    # Candidate status
    # ------------------------------------------------------------------

    @property
    def is_candidate(self) -> bool:
        """Whether this pixel is eligible to be selected as an endmember."""
        return self._is_candidate

    @is_candidate.setter
    def is_candidate(self, value: bool):
        self._is_candidate = bool(value)

    def exclude(self):
        """Mark this pixel as ineligible for endmember selection."""
        self._is_candidate = False

    def include(self):
        """Mark this pixel as eligible for endmember selection."""
        self._is_candidate = True

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.wavelengths)

    def __repr__(self) -> str:
        coord_str = f", centroid={self.centroid}" if self.centroid is not None else ""
        return (
            f"Pixel(id={self.pixel_id!r}, "
            f"n_wavelengths={len(self.wavelengths)}, "
            f"wavelength_unit={self.wavelength_unit!r}, "
            f"spectral_unit={self.spectral_unit!r}"
            f"{coord_str}, "
            f"is_candidate={self.is_candidate})"
        )
