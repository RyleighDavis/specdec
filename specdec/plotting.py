"""
Plotting utilities for spatially resolved spectral abundance maps.

Coordinate convention
---------------------
All pixel coordinates are expected in **positive-degrees-West** (0 W → 360 W)
for longitude and −90 → 90 for latitude.  Internal conversion to the
degrees-East convention required by Cartopy is handled automatically.

Requirements
------------
This module requires ``cartopy`` and ``matplotlib``.  Install via::

    conda install -c conda-forge cartopy matplotlib
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go

import cartopy.crs as ccrs
from shapely.geometry import MultiPolygon, Polygon

from .pixel import Pixel

_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# Div ID used when rendering the progress figure to HTML for Qt injection
_QT_PROGRESS_DIV_ID = "specdec_progress"

# Singleton Qt state — one QApplication + one QWebEngineView for the entire
# process lifetime.  PyQtWebEngine segfaults when a second QWebEngineView is
# created after the first has been destroyed (the Chromium engine tears down).
# Using a single persistent window that navigates between pages sidesteps this.
_qt: dict = {}   # keys: "app", "view"


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------


def _get_or_create_qt_view():
    """
    Return ``(app, view)``, creating the singleton pair on first call.
    Returns ``(None, None)`` if PyQtWebEngine is not available.
    """
    if "view" not in _qt:
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            import sys
            app = QApplication.instance() or QApplication(sys.argv)
            view = QWebEngineView()
            _qt["app"] = app
            _qt["view"] = view
        except ImportError:
            return None, None
    return _qt.get("app"), _qt.get("view")


def _show_plotly_qt_blocking(
    fig: go.Figure,
    title: str = "specdec",
    width: int = 1300,
    height: int = 700,
) -> None:
    """
    Render *fig* in the singleton Qt pop-up window and block (responsively)
    until the user presses Enter in the terminal.

    The window stays open and is reused by the progress tracker afterwards.
    Falls back to ``fig.show()`` if PyQtWebEngine is not available.
    """
    import os, select, sys, tempfile, time as _time
    from PyQt5.QtCore import QUrl

    app, view = _get_or_create_qt_view()
    if view is None:
        warnings.warn(
            "PyQtWebEngine is not installed — falling back to browser display. "
            "Install with: pip install PyQtWebEngine",
            stacklevel=3,
        )
        fig.show()
        return

    html = fig.to_html(include_plotlyjs=True, full_html=True)
    html = html.replace(
        "<head>",
        "<head><style>"
        "html,body{margin:0;padding:0;overflow:hidden;height:100%;}"
        "</style>",
        1,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                      encoding="utf-8")
    tmp.write(html)
    tmp.close()

    view.setWindowTitle(title)
    view.resize(width, height)
    view.load(QUrl.fromLocalFile(tmp.name))
    view.show()

    # Pump Qt events while also watching for Enter on stdin so the window
    # stays responsive (resize, pan, zoom) during the wait.
    print("\nPress Enter to start the decomposition run...", flush=True)
    while True:
        app.processEvents()
        # Non-blocking stdin check (Unix / macOS)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break
        _time.sleep(0.02)

    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    # The window stays open — it will be reused for the progress tracker.


# ---------------------------------------------------------------------------
# Coordinate conversion utilities
# ---------------------------------------------------------------------------


def to_WPos(longitude, num_format="g"):
    """
    Format a longitude (in degrees East) as a positive-degrees-West string.

    Parameters
    ----------
    longitude : float
        Longitude in degrees East (−180 to 180).
    num_format : str
        ``format``-style format code for the numeric part.  Default ``'g'``.

    Returns
    -------
    str
        Formatted string, e.g. ``'90°W'``.
    """
    fmt_string = "{longitude:{num_format}}{degree}{hemisphere}"
    if longitude >= 0:
        longitude = 360 - longitude
    return fmt_string.format(
        longitude=abs(longitude),
        num_format=num_format,
        hemisphere="W",
        degree="\u00b0",
    )


def gl_WPos(ax, fontsize=12, **kwargs):
    """
    Add gridlines to a Cartopy axis with longitude labels in 0 W → 360 W format.

    Parameters
    ----------
    ax : cartopy GeoAxes
        The axis to annotate.
    fontsize : int
        Font size for tick labels.  Default ``12``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.gridlines()``.

    Returns
    -------
    ax : GeoAxes
        The same axis (for chaining).
    gl : cartopy.mpl.gridliner.Gridliner

    Examples
    --------
    ::

        ax, gl = gl_WPos(ax, fontsize=14, linestyle='--', color='gray')
    """
    gl = ax.gridlines(draw_labels=True, **kwargs)
    gl.xformatter = mticker.FuncFormatter(lambda v, pos: to_WPos(v))
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}
    gl.top_labels = False
    gl.right_labels = False
    return ax, gl


def to_EW(longitude):
    """
    Convert longitude from positive-degrees-West [360 W, 0 W] to
    degrees-East [−180 E, 180 E].

    Parameters
    ----------
    longitude : float
        Longitude in positive-degrees-West convention.

    Returns
    -------
    float
    """
    if longitude <= 180:
        return -1.0 * longitude
    return 360.0 - longitude


def to_W(longitude):
    """
    Convert longitude from degrees-East [−180 E, 180 E] to
    positive-degrees-West [0 W, 360 W].

    Parameters
    ----------
    longitude : float
        Longitude in degrees-East convention.

    Returns
    -------
    float
    """
    if longitude >= 0:
        return 360.0 - longitude
    return abs(longitude)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _subplot_grid(n: int) -> Tuple[int, int]:
    """Return ``(nrows, ncols)`` for *n* subplots in a compact layout."""
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def _fix_antimeridian_ring(coords: list) -> list:
    """
    If a ring's x-coordinates span more than 180° the polygon would be drawn
    the wrong way around the globe (antimeridian wrap).  Fix by shifting all
    negative x-values up by 360° so every corner sits on the same side.
    """
    xs = [pt[0] for pt in coords]
    if max(xs) - min(xs) > 180.0:
        coords = [(x + 360.0 if x < 0.0 else x, y) for x, y in coords]
    return coords


def _transform_polygon_west_to_east(polygon: Polygon) -> Polygon:
    """Return a new :class:`~shapely.geometry.Polygon` with x-coordinates
    converted from positive-degrees-West to degrees-East, handling the
    antimeridian correctly."""
    exterior = _fix_antimeridian_ring(
        [(to_EW(x), y) for x, y in polygon.exterior.coords]
    )
    interiors = [
        _fix_antimeridian_ring([(to_EW(x), y) for x, y in ring.coords])
        for ring in polygon.interiors
    ]
    return Polygon(exterior, interiors)


def _transform_geometry_west_to_east(geom):
    """Convert any Polygon or MultiPolygon from W-space to E-space."""
    if isinstance(geom, Polygon):
        return _transform_polygon_west_to_east(geom)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon(
            [_transform_polygon_west_to_east(p) for p in geom.geoms]
        )
    raise TypeError(
        f"Expected Polygon or MultiPolygon, got {type(geom).__name__}."
    )


def _draw_pixel(ax, pixel: Pixel, color, data_crs) -> None:
    """
    Draw *pixel* on *ax* as a filled polygon coloured by *color*.

    If the pixel has no polygon coordinates it is silently skipped.
    Pixels whose ``polygon`` is a
    :class:`~shapely.geometry.MultiPolygon` (antimeridian / prime-meridian
    split) are drawn as separate patches.
    """
    if pixel.polygon is None:
        return

    geom_e = _transform_geometry_west_to_east(pixel.polygon)
    ax.add_geometries(
        [geom_e],
        crs=data_crs,
        facecolor=color,
        edgecolor="none",
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def plot_abundance_map(
    pixels: List[Pixel],
    abundances: np.ndarray,
    endmembers: Optional[List[Pixel]] = None,
    projection: str = "platecarree",
    central_longitude: float = 0.0,
    central_latitude: float = 0.0,
    cmap: str = "plasma",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    gridlines: bool = True,
    gridline_kwargs: Optional[dict] = None,
    colorbar: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot spatial abundance maps — one subplot per endmember.

    Each pixel is rendered as a filled polygon coloured by its abundance
    fraction for that endmember.  Pixels with no coordinate geometry are
    silently skipped.

    Parameters
    ----------
    pixels : list of Pixel
        Ordered list of pixels corresponding to rows of *abundances*.
    abundances : ndarray, shape (n_pixels, n_endmembers)
        Abundance fractions.  Typically ``decomp.abundances``.
    endmembers : list of Pixel, optional
        Pixel objects for the endmembers.  Their ``pixel_id`` is used as the
        subplot title when provided; otherwise titles default to "EM 1", etc.
    projection : {'platecarree', 'orthographic'}
        Map projection.  Default ``'platecarree'``.
    central_longitude : float
        Central meridian for the Orthographic projection in degrees East.
        Default ``0.0``.
    central_latitude : float
        Central latitude for the Orthographic projection.  Default ``0.0``.
    cmap : str
        Matplotlib colormap name.  Default ``'plasma'``.
    vmin, vmax : float
        Colormap range applied uniformly across all subplots.
        Default ``0.0`` / ``1.0``.
    figsize : (float, float), optional
        Figure dimensions in inches.  Auto-computed from the subplot grid if
        not supplied.
    gridlines : bool
        Draw gridlines with positive-degrees-West longitude labels.
        Default ``True``.
    gridline_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`gl_WPos` (which passes
        them on to ``ax.gridlines``).  Common keys: ``fontsize``,
        ``linestyle``, ``color``, ``alpha``.
    colorbar : bool
        Attach a vertical colorbar to each subplot.  Default ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of GeoAxes, shape (n_endmembers,)
        Flat array of axes in endmember order.

    Raises
    ------
    ValueError
        If ``len(pixels) != abundances.shape[0]`` or *projection* is unknown.

    Examples
    --------
    Plot converged endmember abundances on a global PlateCarree map::

        from specdec.plotting import plot_abundance_map

        fig, axes = plot_abundance_map(
            ds.pixels,
            decomp.abundances,
            endmembers=decomp.endmembers,
        )
        plt.show()

    Orthographic view centred at 90 W, 5 N::

        fig, axes = plot_abundance_map(
            ds.pixels,
            decomp.abundances,
            projection='orthographic',
            central_longitude=-90.0,
            central_latitude=5.0,
        )
    """
    abundances = np.asarray(abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            "abundances must be a 2-D array of shape (n_pixels, n_endmembers)."
        )

    n_pixels, n_em = abundances.shape
    if len(pixels) != n_pixels:
        raise ValueError(
            f"len(pixels) ({len(pixels)}) must equal abundances.shape[0] ({n_pixels})."
        )

    # --- Projection ----------------------------------------------------------
    proj_key = projection.lower().replace(" ", "")
    if proj_key == "platecarree":
        map_proj = ccrs.PlateCarree()
    elif proj_key == "orthographic":
        map_proj = ccrs.Orthographic(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
        )
    else:
        raise ValueError(
            f"projection must be 'platecarree' or 'orthographic', got {projection!r}."
        )

    data_crs = ccrs.PlateCarree()

    # --- Layout — one row per endmember, single column -----------------------
    nrows, ncols = n_em, 1
    if figsize is None:
        figsize = (10.0, nrows * 3.5)

    fig, raw_axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": map_proj},
        squeeze=False,
    )
    axes_flat = raw_axes.flatten()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    gl_kw = dict(gridline_kwargs or {})

    # --- Draw ----------------------------------------------------------------
    for j, ax in enumerate(axes_flat):
        if j >= n_em:
            ax.set_visible(False)
            continue

        ax.set_global()

        abund_j = abundances[:, j]
        for i, pixel in enumerate(pixels):
            _draw_pixel(ax, pixel, color=colormap(norm(abund_j[i])), data_crs=data_crs)

        # Endmember stars
        if endmembers is not None:
            for k, em in enumerate(endmembers):
                c = em.centroid
                if c is None:
                    continue
                em_lon, em_lat = float(c[0]), float(c[1])
                color = _COLORS[k % len(_COLORS)]
                ax.plot(
                    em_lon, em_lat,
                    marker="*",
                    markersize=12,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    linestyle="none",
                    transform=data_crs,
                    zorder=5,
                    label=f"EM {k + 1}",
                )

        # Title
        if endmembers is not None and j < len(endmembers):
            em = endmembers[j]
            lon = em.metadata.get("lon")
            lat = em.metadata.get("lat")
            if lon is not None and lat is not None:
                title = f"EM {j + 1} ({float(lon):.1f}°W, {float(lat):.1f}°N)"
            else:
                title = f"EM {j + 1}"
        else:
            title = f"EM {j + 1}"
        ax.set_title(title)

        # Gridlines
        if gridlines:
            try:
                gl_WPos(ax, **gl_kw)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"Gridlines could not be added to subplot {j}: {exc}",
                    stacklevel=2,
                )

        # Colorbar
        if colorbar:
            sm = cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, axes_flat[:n_em]


# ---------------------------------------------------------------------------
# Abundance simplex / ternary plot (Plotly)
# ---------------------------------------------------------------------------


def plot_abundance_simplex(
    abundances: np.ndarray,
    endmembers: List[Pixel],
    em_index_history: list,
    endmember_indices: np.ndarray,
    pixels: Optional[List[Pixel]] = None,
    title: str = "Abundance Simplex",
    show: bool = True,
) -> go.Figure:
    """
    Visualise per-pixel endmember abundances in the mixing simplex.

    For **3 endmembers** a native Plotly ternary diagram is produced.
    For **N > 3 endmembers** a 2-D barycentric projection is used: the N
    corners are placed at equal angles around the unit circle, and each
    pixel's position is the abundance-weighted centroid of those corners.

    Each plot shows:

    * **Grey dots** — every pixel once, at low opacity.
    * **Coloured dashed paths** — only the pixels that were accepted as an
      endmember at some point, connected in the order they were accepted for
      that slot.
    * **Stars** — the final best endmember pixels.

    Parameters
    ----------
    abundances : ndarray, shape (n_pixels, n_endmembers)
        Normalised per-pixel abundance fractions.
    endmembers : list of Pixel
        Best-fit endmember pixel objects.
    em_index_history : list of ndarray
        Endmember global-index arrays at each accepted step.
    endmember_indices : ndarray
        Global indices of the final best endmembers into the pixel list.
    pixels : list of Pixel, optional
        Full pixel list used to populate lon/lat hover labels on path points.
    title : str
        Figure title.
    show : bool
        Display the figure immediately (default ``True``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_em = abundances.shape[1]

    def _coord_label(em: Pixel) -> str:
        lon = em.metadata.get("lon")
        lat = em.metadata.get("lat")
        if lon is not None and lat is not None:
            return f"EM {endmembers.index(em) + 1} ({float(lon):.1f}°W, {float(lat):.1f}°N)"
        return f"EM {endmembers.index(em) + 1}"

    def _em_label(j: int) -> str:
        return _coord_label(endmembers[j])

    def _path_hover(gi: int) -> str:
        if pixels is None:
            return f"pixel {gi}"
        px = pixels[gi]
        lon = px.metadata.get("lon")
        lat = px.metadata.get("lat")
        if lon is not None and lat is not None:
            return f"{float(lon):.1f}°W, {float(lat):.1f}°N"
        return px.pixel_id or f"pixel {gi}"

    # Build unique accepted-pixel path per EM slot (deduplicate consecutive repeats)
    paths: List[List[int]] = [[] for _ in range(n_em)]
    for step_indices in em_index_history:
        for j in range(n_em):
            idx = int(step_indices[j])
            if not paths[j] or paths[j][-1] != idx:
                paths[j].append(idx)

    # ------------------------------------------------------------------ N==3
    if n_em == 3:
        # Draw the ternary diagram on Cartesian axes so we control z-order exactly:
        #   grey pixels → gridlines → triangle border → EM paths → stars
        # EM1 = top, EM2 = bottom-left, EM3 = bottom-right (equilateral triangle).
        T = np.array([
            [ 0.0,             1.0 ],   # EM1 — top
            [-np.sqrt(3) / 2, -0.5],   # EM2 — bottom-left
            [ np.sqrt(3) / 2, -0.5],   # EM3 — bottom-right
        ])

        def _b2c(ab_rows):
            """Barycentric (n,3) → Cartesian (n,2)."""
            return np.asarray(ab_rows) @ T

        fig = go.Figure()

        # 1. Grey pixel cloud (bottom layer)
        xy = _b2c(abundances)
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode="markers",
            name="Pixels",
            marker=dict(size=4, color="lightgray", opacity=0.5,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 2. Gridlines (over grey pixels, under paths)
        grid_levels = [0.2, 0.4, 0.6, 0.8]
        gx, gy = [], []
        for k in grid_levels:
            for em_i in range(3):
                others = [i for i in range(3) if i != em_i]
                p1 = np.zeros(3); p1[em_i] = k; p1[others[0]] = 1 - k
                p2 = np.zeros(3); p2[em_i] = k; p2[others[1]] = 1 - k
                c1, c2 = p1 @ T, p2 @ T
                gx += [c1[0], c2[0], None]
                gy += [c1[1], c2[1], None]
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines",
            line=dict(color="black", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))

        # 3. Triangle border (over gridlines, under paths)
        border = np.vstack([T, T[0]])
        fig.add_trace(go.Scatter(
            x=border[:, 0], y=border[:, 1], mode="lines",
            line=dict(color="black", width=2),
            showlegend=False, hoverinfo="skip",
        ))

        # 4. Accepted EM paths (over grid and border)
        for j in range(n_em):
            color = _COLORS[j % len(_COLORS)]
            ab = abundances[paths[j]]
            xy_path = _b2c(ab)
            hover = [_path_hover(gi) for gi in paths[j]]
            fig.add_trace(go.Scatter(
                x=xy_path[:, 0], y=xy_path[:, 1],
                mode="lines+markers",
                name=_em_label(j),
                line=dict(color=color, width=3, dash="dash"),
                marker=dict(size=9, color=color,
                            line=dict(color="white", width=1.5)),
                text=hover, hoverinfo="text",
            ))

        # 5. Final endmember stars (topmost)
        for j in range(n_em):
            color = _COLORS[j % len(_COLORS)]
            gi = int(endmember_indices[j])
            xy_star = abundances[gi] @ T
            fig.add_trace(go.Scatter(
                x=[xy_star[0]], y=[xy_star[1]],
                mode="markers",
                showlegend=False,
                marker=dict(size=20, color=color, symbol="star",
                            line=dict(color="white", width=1.5)),
            ))

        # Corner axis labels (EM name only; lon/lat stays in legend)
        pad = 1.08
        corner_anchors = [("center", "bottom"), ("right", "top"), ("left", "top")]
        for j in range(3):
            lx, ly = T[j] * pad
            xa, ya = corner_anchors[j]
            fig.add_annotation(
                x=lx, y=ly, text=f"EM {j + 1}",
                showarrow=False, font=dict(size=14),
                xanchor=xa, yanchor=ya,
            )

        # Tick labels along each axis edge
        # EM1=k on left edge (T[1]→T[0]), EM2=k on bottom edge (T[2]→T[1]),
        # EM3=k on right edge (T[0]→T[2])
        tick_cfg = [
            (0, 1, "right",  "middle", -0.05,  0.0),
            (1, 2, "center", "top",     0.0,  -0.05),
            (2, 0, "left",   "middle",  0.05,  0.0),
        ]
        for ei, ej, xa, ya, ox, oy in tick_cfg:
            for k in grid_levels:
                pt = k * T[ei] + (1 - k) * T[ej]
                fig.add_annotation(
                    x=pt[0] + ox, y=pt[1] + oy,
                    text=f"{k:.1f}",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    xanchor=xa, yanchor=ya,
                )

        fig.update_layout(
            xaxis=dict(visible=False, range=[-1.12, 1.12]),
            yaxis=dict(visible=False, range=[-0.75, 1.18],
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top"),
            height=580,
            width=680,
            margin=dict(l=60, r=60, t=40, b=100),
        )

    # ------------------------------------------------------------------ N > 3
    else:
        angles = np.pi / 2 - 2 * np.pi * np.arange(n_em) / n_em
        cx = np.cos(angles)
        cy = np.sin(angles)

        px_x = abundances @ cx
        px_y = abundances @ cy

        fig = go.Figure()

        # Background circle outline
        theta = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode="lines",
            line=dict(color="lightgray", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

        # Simplex edges between corners
        for j in range(n_em):
            k = (j + 1) % n_em
            fig.add_trace(go.Scatter(
                x=[cx[j], cx[k]], y=[cy[j], cy[k]],
                mode="lines",
                line=dict(color="lightgray", width=1),
                showlegend=False, hoverinfo="skip",
            ))

        # All pixels — grey, one dot each
        fig.add_trace(go.Scatter(
            x=px_x, y=px_y,
            mode="markers",
            name="Pixels",
            marker=dict(size=4, color="lightgray", opacity=0.5,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            showlegend=False,
        ))

        # Accepted paths — coloured, dashed
        for j in range(n_em):
            color = _COLORS[j % len(_COLORS)]
            ab = abundances[paths[j]]
            xs = ab @ cx
            ys = ab @ cy
            hover = [_path_hover(gi) for gi in paths[j]]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name=_em_label(j),
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=8, color=color,
                            line=dict(color="white", width=1)),
                text=hover,
                hoverinfo="text",
            ))

        # Final best endmember stars + corner labels
        for j in range(n_em):
            color = _COLORS[j % len(_COLORS)]
            gi = int(endmember_indices[j])
            xs = float(abundances[gi] @ cx)
            ys = float(abundances[gi] @ cy)
            fig.add_trace(go.Scatter(
                x=[xs], y=[ys],
                mode="markers",
                name=f"EM {j+1} best",
                showlegend=False,
                marker=dict(size=18, color=color, symbol="star",
                            line=dict(color="white", width=1.5)),
            ))
            label_scale = 1.15
            fig.add_annotation(
                x=cx[j] * label_scale, y=cy[j] * label_scale,
                text=_em_label(j),
                showarrow=False,
                font=dict(size=12, color=color),
                xanchor="center",
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            template="plotly_white",
            xaxis=dict(visible=False, range=[-1.35, 1.35]),
            yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x"),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05, yanchor="top"),
            height=600,
            width=620,
            margin=dict(l=40, r=40, t=80, b=80),
        )

    if show and not _is_jupyter():
        w = 700 if n_em == 3 else 640
        _show_plotly_qt_blocking(fig, title=title, width=w, height=620)

    return fig


# ---------------------------------------------------------------------------
# Endmember search history plot (Plotly)
# ---------------------------------------------------------------------------


def plot_search_history(
    pixels: List[Pixel],
    em_index_history: list,
    endmember_indices: np.ndarray,
    tried_moves: Optional[list] = None,
    title: str = "Endmember Search History",
    show: bool = True,
) -> go.Figure:
    """
    Plot the endmember search trajectory on a lon/lat map.

    Mirrors the left panel of the live progress figure, but as a standalone
    static plot suitable for post-hoc analysis.

    Parameters
    ----------
    pixels : list of Pixel
        All dataset pixels (used to look up centroids by global index).
    em_index_history : list of ndarray
        Endmember global-index arrays at each accepted step, as stored in
        ``DecompositionResults.em_index_history``.
    endmember_indices : ndarray
        Global indices of the final best endmembers.
    tried_moves : list of (int, int), optional
        All ``(em_position, pixel_global_idx)`` pairs that were evaluated
        during the search, as stored in ``DecompositionResults.tried_moves``.
        If ``None`` or empty, only the accepted trajectory is shown.
    title : str
        Figure title.
    show : bool
        Display the figure immediately (default ``True``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_em = len(endmember_indices)

    def _centroid(idx):
        c = pixels[idx].centroid
        return (c[0], c[1]) if c is not None else (None, None)

    # Tried positions (light scatter, one trace per EM slot)
    tried_lons: List[List[float]] = [[] for _ in range(n_em)]
    tried_lats: List[List[float]] = [[] for _ in range(n_em)]
    if tried_moves:
        for em_pos, px_idx in tried_moves:
            if 0 <= em_pos < n_em:
                lon, lat = _centroid(px_idx)
                if lon is not None:
                    tried_lons[em_pos].append(lon)
                    tried_lats[em_pos].append(lat)

    # Accepted trajectory (connected dots, one trace per EM slot)
    traj_lons: List[List[float]] = [[] for _ in range(n_em)]
    traj_lats: List[List[float]] = [[] for _ in range(n_em)]
    prev = None
    for step_indices in em_index_history:
        for j in range(n_em):
            if prev is None or step_indices[j] != prev[j]:
                lon, lat = _centroid(int(step_indices[j]))
                if lon is not None:
                    traj_lons[j].append(lon)
                    traj_lats[j].append(lat)
        prev = step_indices

    fig = go.Figure()

    # Tried scatter (light, one trace per EM)
    for j in range(n_em):
        color = _COLORS[j % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=tried_lons[j],
            y=tried_lats[j],
            mode="markers",
            name=f"EM {j+1} tried",
            legendgroup=f"em{j}",
            showlegend=bool(tried_moves),
            marker=dict(size=4, color=color, opacity=0.15),
        ))

    # Accepted trajectory (solid, prominent — rendered after tried scatter so it sits on top)
    for j in range(n_em):
        color = _COLORS[j % len(_COLORS)]
        em = pixels[int(endmember_indices[j])]
        em_label = f"EM {j+1}" if em.pixel_id is None else f"EM {j+1} — {em.pixel_id}"
        fig.add_trace(go.Scatter(
            x=traj_lons[j],
            y=traj_lats[j],
            mode="lines+markers",
            name=em_label,
            legendgroup=f"em{j}",
            line=dict(color=color, width=3),
            marker=dict(size=10, color=color,
                        line=dict(color="white", width=1)),
        ))

    # Best-position stars
    for j in range(n_em):
        color = _COLORS[j % len(_COLORS)]
        lon, lat = _centroid(int(endmember_indices[j]))
        em = pixels[int(endmember_indices[j])]
        em_label = f"EM {j+1}" if em.pixel_id is None else f"EM {j+1} — {em.pixel_id}"
        fig.add_trace(go.Scatter(
            x=[lon] if lon is not None else [],
            y=[lat] if lat is not None else [],
            mode="markers+text",
            name=f"EM {j+1} best",
            legendgroup=f"em{j}",
            showlegend=False,
            marker=dict(size=18, color=color, symbol="star",
                        line=dict(color="white", width=1.5)),
            text=[f"EM {j+1}"],
            textposition="top center",
            textfont=dict(size=11),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        xaxis=dict(title="Longitude (°W)", range=[360, 0]),
        yaxis=dict(title="Latitude (°N)", range=[-90, 90]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top"),
        height=500,
        width=800,
        margin=dict(l=60, r=40, t=60, b=80),
    )

    if show and not _is_jupyter():
        _show_plotly_qt_blocking(fig, title=title, width=820, height=520)

    return fig


# ---------------------------------------------------------------------------
# Interactive spectral explorer (Plotly + background HTTP server)
# ---------------------------------------------------------------------------


def _start_explorer_server(pixels, em_spectra, abundances, scale_factors, rms_errors):
    """
    Start a background HTTP server that serves per-pixel spectral data on demand.
    Returns the port number.  The server runs as a daemon thread and dies with
    the Python process.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    # Build (lon_int, lat_int) → pixel-index lookup
    coord_to_gi: dict = {}
    for gi, px in enumerate(pixels):
        lon = px.metadata.get("lon")
        lat = px.metadata.get("lat")
        if lon is not None and lat is not None:
            coord_to_gi[(int(lon), int(lat))] = gi

    n_em = len(em_spectra)
    _state = {
        "pixels":        pixels,
        "em_spectra":    em_spectra,
        "abundances":    abundances,
        "scale_factors": scale_factors,
        "rms_errors":    rms_errors,
        "coord_to_gi":   coord_to_gi,
        "n_em":          n_em,
        "html":          None,   # set by caller after HTML is built
    }

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _state["html"]
                if body is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/px":
                qs = parse_qs(parsed.query)
                try:
                    lon = int(qs["lon"][0])
                    lat = int(qs["lat"][0])
                except (KeyError, ValueError, IndexError):
                    self._send(400, {"error": "bad request"})
                    return
                gi = _state["coord_to_gi"].get((lon, lat))
                if gi is None:
                    self._send(404, {"error": "pixel not found"})
                    return
                px   = _state["pixels"][gi]
                sf   = float(_state["scale_factors"][gi]) if _state["scale_factors"] is not None else 1.0
                ab   = _state["abundances"][gi]
                ems  = _state["em_spectra"]
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    mod      = sf * (ab @ ems)
                    contribs = [(sf * float(ab[j]) * ems[j]).tolist()
                                for j in range(_state["n_em"])]
                self._send(200, {
                    "gi":       int(gi),
                    "obs":      [round(float(v), 5) for v in px.spectrum],
                    "mod":      [round(float(v), 5) for v in mod],
                    "contribs": [[round(float(v), 5) for v in c] for c in contribs],
                    "ab":       [round(float(v), 5) for v in ab],
                    "rms":      round(float(_state["rms_errors"][gi]), 6)
                                if _state["rms_errors"] is not None else None,
                    "id":       px.pixel_id or f"pixel_{gi}",
                    "lat":      lat,
                    "lon":      lon,
                })
            elif parsed.path == "/ping":
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def _send(self, code, data):
            body = _json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # silence request logs

    # Pick a free port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Wait until the server is ready
    for _ in range(30):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)

    return port, _state


def plot_interactive_explorer(
    pixels: List[Pixel],
    wavelengths: np.ndarray,
    abundances: np.ndarray,
    endmembers: List[Pixel],
    endmember_indices: Optional[np.ndarray] = None,
    scale_factors: Optional[np.ndarray] = None,
    rms_errors: Optional[np.ndarray] = None,
    title: str = "Spectral Explorer",
    show: bool = True,
) -> go.Figure:
    """
    Interactive two-panel explorer: a global map on the left (click any cell
    to select it) and a spectrum panel on the right.

    The map is a heatmap of the dominant endmember at each 1°×1° grid cell,
    so every cell is clearly visible and directly clickable regardless of zoom
    level.  Clicking a cell fetches that pixel's spectra on demand from a
    lightweight background server — there is no upfront serialisation of all
    pixel data, so the figure renders immediately.

    Parameters
    ----------
    pixels : list of Pixel
        All dataset pixels in the same order as rows of *abundances*.
    wavelengths : ndarray
        Wavelength axis shared by all pixel spectra.
    abundances : ndarray, shape (n_pixels, n_endmembers)
        Normalised per-pixel abundance fractions.
    endmembers : list of Pixel
        Best-fit endmember pixel objects.
    endmember_indices : ndarray, optional
        Global indices of the endmembers into *pixels*.
    scale_factors : ndarray, shape (n_pixels,), optional
        Per-pixel scale factors (``free_sum`` mode).
    rms_errors : ndarray, shape (n_pixels,), optional
        Per-pixel RMS residuals.
    title : str
        Main figure title.
    show : bool
        Open the figure immediately (default ``True``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from plotly.subplots import make_subplots
    import json as _json

    n_em = len(endmembers)
    em_spectra = np.array([em.spectrum for em in endmembers], dtype=float)
    wl = np.asarray(wavelengths, dtype=float)

    # Axis labels
    wl_label = (
        f"Wavelength ({endmembers[0].wavelength_unit})"
        if endmembers and endmembers[0].wavelength_unit else "Wavelength"
    )
    spec_label = (
        endmembers[0].spectral_unit
        if endmembers and endmembers[0].spectral_unit else "Reflectance"
    )

    # ------------------------------------------------------------------
    # Build dominant-EM heatmap grid  (180 rows × 360 cols)
    # x axis: cell-centre longitudes  [0.5, 1.5, …, 359.5]  (°W)
    # y axis: cell-centre latitudes   [-89.5, …, 89.5]       (°N)
    # clicking a cell gives pt.x = lon+0.5, pt.y = lat+0.5;
    # floor() recovers the integer SW-corner lon/lat stored in metadata.
    # ------------------------------------------------------------------
    dominant = np.argmax(abundances, axis=1)
    z_grid = np.full((180, 360), np.nan)
    for gi, px in enumerate(pixels):
        lon_m = px.metadata.get("lon")
        lat_m = px.metadata.get("lat")
        if lon_m is None or lat_m is None:
            continue
        li, ci = int(lat_m) + 90, int(lon_m)
        if 0 <= li < 180 and 0 <= ci < 360:
            z_grid[li, ci] = float(dominant[gi])

    x_lons = [i + 0.5 for i in range(360)]
    y_lats = [i - 89.5 for i in range(180)]

    # Discrete colorscale: equal-width band per endmember
    colorscale = []
    for i in range(n_em):
        lo, hi = i / n_em, (i + 1) / n_em
        colorscale += [[lo, _COLORS[i % len(_COLORS)]], [hi, _COLORS[i % len(_COLORS)]]]

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.45, 0.55],
        subplot_titles=["Dominant Endmember  (click to select)", "Click a pixel on the map"],
        horizontal_spacing=0.08,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 13

    # Trace 0 — heatmap
    fig.add_trace(go.Heatmap(
        z=z_grid,
        x=x_lons,
        y=y_lats,
        colorscale=colorscale,
        zmin=-0.5,
        zmax=n_em - 0.5,
        showscale=False,
        hovertemplate="(%{x:.0f}°W, %{y:.0f}°N)<extra></extra>",
        name="",
    ), row=1, col=1)

    # Traces 1…n_em — endmember stars
    for j, em in enumerate(endmembers):
        color = _COLORS[j % len(_COLORS)]
        lon_m = em.metadata.get("lon")
        lat_m = em.metadata.get("lat")
        em_x = [float(lon_m) + 0.5] if lon_m is not None else []
        em_y = [float(lat_m) + 0.5] if lat_m is not None else []
        lon_label = f"{float(lon_m):.1f}°W" if lon_m is not None else "?"
        lat_label = f"{float(lat_m):.1f}°N" if lat_m is not None else "?"
        fig.add_trace(go.Scatter(
            x=em_x, y=em_y,
            mode="markers+text",
            name=f"EM {j+1} ({lon_label}, {lat_label})",
            marker=dict(size=18, color=color, symbol="star",
                        line=dict(color="white", width=1.5)),
            text=[f"EM {j+1}"],
            textposition="top center",
            textfont=dict(size=10, color=color),
            hoverinfo="name",
            showlegend=True,
        ), row=1, col=1)

    # Trace n_em+1 — selected cell highlight (square, starts empty)
    _T_SEL = n_em + 1
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="markers",
        name="Selected",
        marker=dict(size=14, color="rgba(0,0,0,0)", symbol="square-open",
                    line=dict(color="white", width=2.5)),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # Trace n_em+2 — observed spectrum
    _T_OBS = n_em + 2
    fig.add_trace(go.Scatter(
        x=list(wl), y=[],
        mode="lines", name="Observed",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=2)

    # Trace n_em+3 — modelled spectrum
    _T_MOD = n_em + 3
    fig.add_trace(go.Scatter(
        x=list(wl), y=[],
        mode="lines", name="Modelled",
        line=dict(color="black", width=2),
    ), row=1, col=2)

    # Traces n_em+4…2*n_em+3 — EM contributions
    _T_C0 = n_em + 4
    for j, em in enumerate(endmembers):
        color = _COLORS[j % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=list(wl), y=[],
            mode="lines",
            name=f"EM {j+1}",
            legendgroup=f"em{j}",
            showlegend=True,
            line=dict(color=color, width=1.5, dash="dot"),
            opacity=0.8,
        ), row=1, col=2)

    # Outline trace — thin black lines around each pixel footprint
    outline_x: list = []
    outline_y: list = []
    for px in pixels:
        if px.polygon is None:
            continue
        polys = list(px.polygon.geoms) if hasattr(px.polygon, "geoms") else [px.polygon]
        for poly in polys:
            xs = [c[0] for c in poly.exterior.coords]
            ys = [c[1] for c in poly.exterior.coords]
            outline_x += xs + [None]
            outline_y += ys + [None]
    if outline_x:
        fig.add_trace(go.Scatter(
            x=outline_x, y=outline_y,
            mode="lines",
            line=dict(color="black", width=0.5),
            showlegend=False,
            hoverinfo="skip",
            name="",
        ), row=1, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        height=560, width=1300,
        margin=dict(l=60, r=40, t=80, b=60),
        uirevision="explorer",
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top"),
    )
    fig.update_xaxes(title_text="Longitude (°W)", range=[360, 0], row=1, col=1)
    fig.update_yaxes(title_text="Latitude (°N)", range=[-90, 90], row=1, col=1)
    fig.update_xaxes(title_text=wl_label, row=1, col=2)
    fig.update_yaxes(title_text=spec_label, row=1, col=2)

    # ------------------------------------------------------------------
    # Start background server and build HTML
    # ------------------------------------------------------------------
    port, _srv_state = _start_explorer_server(pixels, em_spectra, abundances, scale_factors, rms_errors)

    # Small constants embedded in JS — no pixel spectra pre-serialised
    wl_js     = _json.dumps([round(float(v), 4) for v in wl])
    colors_js = _json.dumps([_COLORS[j % len(_COLORS)] for j in range(n_em)])
    init_js   = (
        f"var _PORT={port};"
        f"var _WL={wl_js};"
        f"var _N_EM={int(n_em)};"
        f"var _T_SEL={int(_T_SEL)};"
        f"var _T_OBS={int(_T_OBS)};"
        f"var _T_MOD={int(_T_MOD)};"
        f"var _T_C0={int(_T_C0)};"
        f"var _EM_COLORS={colors_js};"
    )

    div_id = "specexplorer"
    handler_js = (
        "(function(){"
        "function _attach(){"
        f"var div=document.getElementById('{div_id}');"
        "if(!div||!div._fullData){{setTimeout(_attach,200);return;}}"
        "div.on('plotly_click',function(evt){"
        "if(!evt||!evt.points||!evt.points[0])return;"
        "var pt=evt.points[0];"
        # Only handle clicks from the map panel (xaxis='x'), ignore right panel (xaxis='x2')
        "if(pt.fullData.xaxis==='x2')return;"
        "var lon=Math.floor(pt.x);"
        "var lat=Math.floor(pt.y);"
        "fetch('http://127.0.0.1:'+_PORT+'/px?lon='+lon+'&lat='+lat)"
        ".then(function(r){return r.ok?r.json():null;})"
        ".then(function(data){"
        "if(!data||data.error)return;"
        # Move highlight square to selected cell centre
        "Plotly.restyle(div,{x:[[lon+0.5]],y:[[lat+0.5]]}, [_T_SEL]);"
        # Observed line: colour by dominant EM
        "var dom=data.ab.indexOf(Math.max.apply(null,data.ab));"
        "Plotly.restyle(div,{y:[data.obs],'line.color':_EM_COLORS[dom]},[_T_OBS]);"
        "Plotly.restyle(div,{y:[data.mod]},[_T_MOD]);"
        "for(var j=0;j<_N_EM;j++){"
        "Plotly.restyle(div,{y:[data.contribs[j]]},[_T_C0+j]);}"
        "var ab=data.ab.map(function(a,j){"
        "return 'EM'+(j+1)+': '+(a*100).toFixed(1)+'%';}).join('  ');"
        "var rms=data.rms!==null?'  RMS='+data.rms.toFixed(4):'';"
        "var lbl=data.id+'  ('+data.lon+'\u00b0W, '+data.lat+'\u00b0N)'+rms+'  |  '+ab;"
        "Plotly.relayout(div,{'annotations[1].text':lbl});"
        "})"
        ".catch(function(e){console.warn('explorer fetch:',e);});"
        "});}"
        "_attach();"
        "})();"
    )

    plotlyjs = "cdn" if _is_jupyter() else True
    html = fig.to_html(include_plotlyjs=plotlyjs, full_html=True, div_id=div_id)
    html = html.replace(
        "<head>",
        "<head><style>html,body{margin:0;padding:0;overflow:hidden;height:100%;}</style>",
        1,
    )
    html = html.replace(
        "</body>",
        f"<script>{init_js}</script>\n<script>{handler_js}</script>\n</body>",
        1,
    )

    # Register HTML with the server (makes http://127.0.0.1:{port}/ serve it)
    _srv_state["html"] = html.encode("utf-8")

    if show:
        if _is_jupyter():
            from IPython.display import display, IFrame
            display(IFrame(src=f"http://127.0.0.1:{port}/", width="100%", height=580))
            return None   # prevent Jupyter auto-rendering the fig a second time
        else:
            import tempfile, webbrowser
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            webbrowser.open(f"file://{tmp.name}")

    return fig


# ---------------------------------------------------------------------------
# Endmember spectra plot (Plotly)
# ---------------------------------------------------------------------------


def plot_endmember_spectra(
    endmembers: List[Pixel],
    cluster_centers: Optional[np.ndarray] = None,
    title: str = "Endmember Spectra",
    show: bool = True,
) -> go.Figure:
    """
    Plot endmember spectra using Plotly, with optional K-means cluster centres
    overlaid for comparison.

    Parameters
    ----------
    endmembers : list of Pixel
        Endmember pixels to plot.  Each pixel's ``wavelengths`` and
        ``spectrum`` are used.
    cluster_centers : ndarray, shape (n_endmembers, n_wavelengths), optional
        K-means cluster centre spectra.  If provided, plotted as dashed lines
        of the same colour as the corresponding endmember.  The wavelength axis
        is taken from the first endmember.
    title : str
        Figure title.  Default ``'Endmember Spectra'``.
    show : bool
        Call ``fig.show()`` before returning.  Default ``True``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for j, em in enumerate(endmembers):
        color = _COLORS[j % len(_COLORS)]
        lon = em.metadata.get("lon")
        lat = em.metadata.get("lat")
        if lon is not None and lat is not None:
            label = f"EM {j + 1} — {float(lon):.1f}°W, {float(lat):.1f}°N"
        else:
            label = f"EM {j + 1}" if em.pixel_id is None else f"EM {j + 1} — {em.pixel_id}"

        fig.add_trace(go.Scatter(
            x=em.wavelengths,
            y=em.spectrum,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    if cluster_centers is not None:
        cluster_centers = np.asarray(cluster_centers, dtype=float)
        # Use wavelengths from the first endmember for the x-axis
        wl = endmembers[0].wavelengths if endmembers else None
        for j, center in enumerate(cluster_centers):
            color = _COLORS[j % len(_COLORS)]
            fig.add_trace(go.Scatter(
                x=wl,
                y=center,
                mode="lines",
                name=f"K-means center {j + 1}",
                line=dict(color=color, width=1.5, dash="dash"),
                opacity=0.6,
            ))

    x_label = (
        f"Wavelength ({endmembers[0].wavelength_unit})"
        if endmembers and endmembers[0].wavelength_unit
        else "Wavelength"
    )
    y_label = (
        endmembers[0].spectral_unit
        if endmembers and endmembers[0].spectral_unit
        else "Reflectance"
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(orientation="v", x=1.01, xanchor="left"),
        template="plotly_white",
    )

    if show and not _is_jupyter():
        _show_plotly_qt_blocking(fig, title=title)

    return fig


# ---------------------------------------------------------------------------
# Live decomposition progress tracker  (Jupyter + terminal)
# ---------------------------------------------------------------------------


def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in (
            "ZMQInteractiveShell",  # Jupyter notebook / lab
            "Shell",                # embedded IPython
        )
    except ImportError:
        return False


class ProgressTracker:
    """
    Container returned by :func:`create_progress_tracker`.

    Holds the Plotly figure and the display mode so that
    :func:`update_progress_tracker` can handle both Jupyter (in-place
    ``FigureWidget`` updates) and terminal (HTML file written on every
    refresh, auto-reloaded by the browser) transparently.

    Attributes
    ----------
    fig : go.FigureWidget or go.Figure
    mode : str  ``'jupyter'`` or ``'terminal'``
    html_path : str or None  Path to the HTML file written in terminal mode.
    n_em : int
    total_combos : int
    patience : int
    """

    __slots__ = ("fig", "mode", "html_path", "n_em", "total_combos", "patience",
                 "qt_app", "qt_view",
                 "_n_tried_sent", "_traj_pts_sent", "_n_rms_sent")

    def __init__(self, fig, mode: str, html_path: Optional[str],
                 n_em: int, total_combos: int, patience: int,
                 qt_app=None, qt_view=None):
        self.fig = fig
        self.mode = mode
        self.html_path = html_path
        self.n_em = n_em
        self.total_combos = total_combos
        self.patience = patience
        self.qt_app = qt_app
        self.qt_view = qt_view
        # How many data points have already been pushed to the Qt window.
        # Used by the incremental JS-injection path so only new points are sent.
        self._n_tried_sent: int = 0
        self._traj_pts_sent: List[int] = [0] * n_em
        self._n_rms_sent: int = 0


def _build_progress_figure(n_em: int, patience: int, total_combos: int) -> go.Figure:
    """Construct the initial (empty) progress figure shared by both modes."""
    from plotly.subplots import make_subplots

    specs = [
        [{"rowspan": 2, "type": "xy"}, {"type": "xy"}],
        [None,                          {"type": "xy"}],
    ]
    fig = make_subplots(
        rows=2, cols=2,
        specs=specs,
        column_widths=[0.55, 0.45],
        row_heights=[0.58, 0.42],
        subplot_titles=[
            "Endmember Search History",
            "Total RMS",
            "Convergence",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    for ann in fig.layout.annotations:
        ann.font.size = 14

    # MAP — tried (non-accepted) positions: light scatter, one per EM slot
    for j in range(n_em):
        color = _COLORS[j % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode="markers",
            name=f"EM {j + 1} tried",
            legendgroup=f"em{j}",
            showlegend=False,
            marker=dict(size=5, color=color, opacity=0.25),
        ), row=1, col=1)

    # MAP — accepted trajectory: darker colour, dotted connecting line
    for j in range(n_em):
        color = _COLORS[j % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode="lines+markers",
            name=f"EM {j + 1} path",
            legendgroup=f"em{j}",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(size=7, color=color),
        ), row=1, col=1)

    # MAP — current-best star markers (one per EM slot)
    for j in range(n_em):
        color = _COLORS[j % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode="markers+text",
            name=f"EM {j + 1} best",
            legendgroup=f"em{j}",
            showlegend=False,
            marker=dict(size=18, color=color, symbol="star",
                        line=dict(color="white", width=1.5)),
            text=[f"EM {j + 1}"],
            textposition="top center",
            textfont=dict(size=11),
        ), row=1, col=1)

    # RMS history scatter
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="lines+markers",
        name="Total RMS",
        showlegend=False,
        line=dict(color="#636EFA", width=2),
        marker=dict(size=5, color="#636EFA"),
    ), row=1, col=2)

    # Convergence bars (patience + RMS stability)
    fig.add_trace(go.Bar(
        y=["Patience", "RMS stability"],
        x=[0, 0],
        orientation="h",
        marker_color=["#EF553B", "#00CC96"],
        text=[f"0 / {patience}", "N/A"],
        textposition="outside",
        showlegend=False,
        width=[0.5, 0.5],
    ), row=2, col=2)

    # Dashed vertical line at 100 % — the target for both bars
    fig.add_shape(
        type="line",
        x0=100, x1=100, y0=-0.5, y1=1.5,
        line=dict(color="black", width=1.5, dash="dash"),
        row=2, col=2,
    )

    fig.update_layout(
        width=1280,
        height=620,
        margin=dict(l=60, r=40, t=60, b=90),
        title=dict(
            text="EndmemberDecomposition — Live Progress",
            font=dict(size=14),
            x=0.0,
            xanchor="left",
        ),
        template="plotly_white",
        uirevision="constant",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
            yanchor="top",
        ),
    )
    fig.update_xaxes(title_text="Longitude (°W)", range=[360, 0], row=1, col=1)
    fig.update_yaxes(title_text="Latitude (°N)", range=[-90, 90], row=1, col=1)
    fig.update_xaxes(title_text="Accepted step", row=1, col=2)
    fig.update_yaxes(title_text="Total RMS", row=1, col=2)
    fig.update_xaxes(title_text="% of target", range=[0, 105], row=2, col=2)

    return fig


def _write_progress_html(fig: go.Figure, path: str,
                         refresh_seconds: int = 5) -> None:
    """Write *fig* to *path* as a self-refreshing HTML page."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    # Insert an auto-refresh meta tag so the browser reloads periodically
    html = html.replace(
        "<head>",
        f'<head><meta http-equiv="refresh" content="{refresh_seconds}">',
        1,
    )
    with open(path, "w") as fh:
        fh.write(html)


def _apply_tracker_updates(fig, decomp, n_em: int,
                           total_combos: int, patience: int) -> None:
    """
    Write current *decomp* state into *fig*'s traces and annotations.

    Works for both ``go.Figure`` (terminal) and ``go.FigureWidget`` (Jupyter)
    by using the ``.update()`` method on trace and layout objects, which is
    supported by both figure types.
    """
    history = decomp._em_index_history
    rms_hist = decomp._accepted_rms_history
    all_pixels = decomp._all_pixels
    n_visited = len(decomp._visited)

    def _centroid(idx):
        c = all_pixels[idx].centroid
        return (c[0], c[1]) if c is not None else (None, None)

    # Tried positions (all candidates ever sampled) by EM slot
    tried_lons: List[List[float]] = [[] for _ in range(n_em)]
    tried_lats: List[List[float]] = [[] for _ in range(n_em)]
    for em_pos, px_idx in decomp._tried_moves:
        if 0 <= em_pos < n_em:
            lon, lat = _centroid(px_idx)
            if lon is not None:
                tried_lons[em_pos].append(lon)
                tried_lats[em_pos].append(lat)

    # Accepted trajectory (positions where each EM slot actually changed)
    traj_lons: List[List[float]] = [[] for _ in range(n_em)]
    traj_lats: List[List[float]] = [[] for _ in range(n_em)]
    best_lons: List[Optional[float]] = [None] * n_em
    best_lats: List[Optional[float]] = [None] * n_em

    prev = None
    for step_indices in history:
        for j in range(n_em):
            if prev is None or step_indices[j] != prev[j]:
                lon, lat = _centroid(step_indices[j])
                if lon is not None:
                    traj_lons[j].append(lon)
                    traj_lats[j].append(lat)
        prev = step_indices

    if decomp._best_em_indices is not None:
        for j in range(n_em):
            best_lons[j], best_lats[j] = _centroid(int(decomp._best_em_indices[j]))

    # RMS history
    rms_x = list(range(len(rms_hist)))
    rms_y = list(rms_hist)

    # Convergence bar values
    patience_pct = min(decomp._n_no_improvement / patience * 100, 100) if patience else 0
    patience_color = "#2ca02c" if patience_pct >= 100 else "#EF553B"
    patience_text = f"{decomp._n_no_improvement:,} / {patience:,}"

    window = rms_hist[-decomp.rms_history_window:]
    if len(window) >= 2:
        mean_w = float(np.mean(window))
        var_w = (max(window) - min(window)) / mean_w if mean_w > 0 else float("inf")
        stab_pct = min(decomp.rms_tolerance / max(var_w, 1e-12) * 100, 100)
        stab_color = "#2ca02c" if stab_pct >= 100 else "#00CC96"
        stab_text = f"var={var_w:.4f} / tol={decomp.rms_tolerance:.4f}"
    else:
        stab_pct, stab_color = 0.0, "#00CC96"
        stab_text = f"N/A (need ≥ {decomp.rms_history_window} accepted)"

    # Trace layout:
    #   0   .. n_em-1      tried scatter (light)
    #   n_em .. 2n_em-1    accepted trajectory (dark, connected)
    #   2n_em .. 3n_em-1   best-position stars
    #   3n_em              RMS history
    #   3n_em+1            convergence bars

    for j in range(n_em):
        fig.data[j].update(x=tried_lons[j], y=tried_lats[j])

    for j in range(n_em):
        fig.data[n_em + j].update(x=traj_lons[j], y=traj_lats[j])

    for j in range(n_em):
        lon, lat = best_lons[j], best_lats[j]
        fig.data[2 * n_em + j].update(
            x=[lon] if lon is not None else [],
            y=[lat] if lat is not None else [],
            text=[f"EM {j + 1}"] if lon is not None else [],
        )

    fig.data[3 * n_em].update(x=rms_x, y=rms_y)

    fig.data[3 * n_em + 1].update(
        x=[patience_pct, stab_pct],
        text=[patience_text, stab_text],
        marker=dict(color=[patience_color, stab_color]),
    )

    status = "CONVERGED" if decomp.is_converged else "running"
    best_rms_str = (f"best RMS = {decomp._best_total_rms:.6g}"
                    if decomp._best_total_rms is not None else "")
    fig.layout.title.text = (
        f"EndmemberDecomposition — Live Progress"
        f"  |  combos: {n_visited:,} / {total_combos:.2e}"
        f"  |  {best_rms_str}  |  {status}"
    )


def _build_progress_js(decomp, n_em: int, total_combos: int,
                       patience: int, div_id: str) -> str:
    """
    Build a string of Plotly JS commands (``Plotly.restyle`` /
    ``Plotly.relayout``) that update the progress figure in-place.

    Intended for use with ``QWebEnginePage.runJavaScript()``.
    """
    import json

    history = decomp._em_index_history
    rms_hist = decomp._accepted_rms_history
    all_pixels = decomp._all_pixels
    n_visited = len(decomp._visited)

    def _centroid(idx):
        c = all_pixels[idx].centroid
        return (c[0], c[1]) if c is not None else (None, None)

    traj_lons: List[List] = [[] for _ in range(n_em)]
    traj_lats: List[List] = [[] for _ in range(n_em)]
    best_lons: List[Optional[float]] = [None] * n_em
    best_lats: List[Optional[float]] = [None] * n_em

    prev = None
    for step_indices in history:
        for j in range(n_em):
            if prev is None or step_indices[j] != prev[j]:
                lon, lat = _centroid(step_indices[j])
                if lon is not None:
                    traj_lons[j].append(lon)
                    traj_lats[j].append(lat)
        prev = step_indices

    if decomp._best_em_indices is not None:
        for j in range(n_em):
            best_lons[j], best_lats[j] = _centroid(int(decomp._best_em_indices[j]))

    rms_x = list(range(len(rms_hist)))
    rms_y = list(rms_hist)

    patience_pct = min(decomp._n_no_improvement / patience * 100, 100) if patience else 0
    patience_color = "#2ca02c" if patience_pct >= 100 else "#EF553B"
    patience_text = f"{decomp._n_no_improvement:,} / {patience:,}"

    window = rms_hist[-decomp.rms_history_window:]
    if len(window) >= 2:
        mean_w = float(np.mean(window))
        var_w = (max(window) - min(window)) / mean_w if mean_w > 0 else float("inf")
        stab_pct = min(decomp.rms_tolerance / max(var_w, 1e-12) * 100, 100)
        stab_color = "#2ca02c" if stab_pct >= 100 else "#00CC96"
        stab_text = f"var={var_w:.4f} / tol={decomp.rms_tolerance:.4f}"
    else:
        stab_pct, stab_color = 0.0, "#00CC96"
        stab_text = f"N/A (need ≥ {decomp.rms_history_window} accepted)"

    status = "CONVERGED" if decomp.is_converged else "running"
    best_rms_str = (f"best RMS = {decomp._best_total_rms:.6g}"
                    if decomp._best_total_rms is not None else "")
    annotation_text = (
        f"Convergence Progress  "
        f"[combos: {n_visited:,} / {total_combos:.2e}"
        f"  |  {best_rms_str}  |  {status}]"
    )

    d = json.dumps  # shorthand
    cmds: List[str] = []

    # Trajectory traces (indices 0 .. n_em-1)
    for j in range(n_em):
        cmds.append(
            f"Plotly.restyle({d(div_id)}, "
            f"{{x: {d([traj_lons[j]])}, y: {d([traj_lats[j]])}}}, [{j}]);"
        )

    # Best-position star markers (indices n_em .. 2*n_em-1)
    for j in range(n_em):
        lon, lat = best_lons[j], best_lats[j]
        x_val = [lon] if lon is not None else []
        y_val = [lat] if lat is not None else []
        text_val = [f"EM {j + 1}"] if lon is not None else []
        cmds.append(
            f"Plotly.restyle({d(div_id)}, "
            f"{{x: {d([x_val])}, y: {d([y_val])}, text: {d([text_val])}}}, "
            f"[{n_em + j}]);"
        )

    # RMS history scatter (index 2*n_em)
    cmds.append(
        f"Plotly.restyle({d(div_id)}, "
        f"{{x: {d([rms_x])}, y: {d([rms_y])}}}, [{2 * n_em}]);"
    )

    # Convergence bars (index 2*n_em+1)
    cmds.append(
        f"Plotly.restyle({d(div_id)}, "
        f"{{x: {d([[patience_pct, stab_pct]])}, "
        f"text: {d([[patience_text, stab_text]])}, "
        f"'marker.color': {d([[patience_color, stab_color]])}}}, "
        f"[{2 * n_em + 1}]);"
    )

    # Annotation text
    cmds.append(
        f"Plotly.relayout({d(div_id)}, "
        f"{{'annotations[2].text': {d(annotation_text)}}});"
    )

    return "\n".join(cmds)


def _build_incremental_js(decomp, tracker: ProgressTracker, div_id: str) -> str:
    """
    Build JS that adds only the *new* points to the live Plotly figure using
    ``Plotly.extendTraces`` (for accumulating series) and ``Plotly.restyle``
    (for single-value updates like the best-position stars and convergence bars).

    Trace layout expected in the figure
    ------------------------------------
    0   .. n_em-1      tried scatter (light)
    n_em .. 2n_em-1    accepted trajectory (dark, connected)
    2n_em .. 3n_em-1   best-position stars
    3n_em              RMS history
    3n_em+1            convergence bars
    """
    import json

    n_em = tracker.n_em
    patience = tracker.patience
    total_combos = tracker.total_combos
    all_pixels = decomp._all_pixels
    rms_hist = decomp._accepted_rms_history
    history = decomp._em_index_history

    def _centroid(idx):
        c = all_pixels[idx].centroid
        return (c[0], c[1]) if c is not None else (None, None)

    d = json.dumps
    cmds: List[str] = []

    # ── 1. New tried positions (extendTraces) ─────────────────────────────
    new_tried = decomp._tried_moves[tracker._n_tried_sent:]
    if new_tried:
        new_tried_lons: List[List[float]] = [[] for _ in range(n_em)]
        new_tried_lats: List[List[float]] = [[] for _ in range(n_em)]
        for em_pos, px_idx in new_tried:
            if 0 <= em_pos < n_em:
                lon, lat = _centroid(px_idx)
                if lon is not None:
                    new_tried_lons[em_pos].append(lon)
                    new_tried_lats[em_pos].append(lat)
        for j in range(n_em):
            if new_tried_lons[j]:
                cmds.append(
                    f"Plotly.extendTraces({d(div_id)}, "
                    f"{{x:{d([new_tried_lons[j]])},y:{d([new_tried_lats[j]])}}},{[j]});"
                )

    # ── 2. New accepted trajectory points (extendTraces) ─────────────────
    # Recompute full trajectory to find the new points since last send.
    traj_lons: List[List[float]] = [[] for _ in range(n_em)]
    traj_lats: List[List[float]] = [[] for _ in range(n_em)]
    prev = None
    for step_indices in history:
        for j in range(n_em):
            if prev is None or step_indices[j] != prev[j]:
                lon, lat = _centroid(step_indices[j])
                if lon is not None:
                    traj_lons[j].append(lon)
                    traj_lats[j].append(lat)
        prev = step_indices

    new_traj_pts: List[int] = []
    for j in range(n_em):
        new_lons = traj_lons[j][tracker._traj_pts_sent[j]:]
        new_lats = traj_lats[j][tracker._traj_pts_sent[j]:]
        new_traj_pts.append(len(traj_lons[j]))
        if new_lons:
            cmds.append(
                f"Plotly.extendTraces({d(div_id)}, "
                f"{{x:{d([new_lons])},y:{d([new_lats])}}},{[n_em + j]});"
            )

    # ── 3. Best-position stars (restyle — always the current best) ────────
    if decomp._best_em_indices is not None:
        for j in range(n_em):
            lon, lat = _centroid(int(decomp._best_em_indices[j]))
            x_val = [lon] if lon is not None else []
            y_val = [lat] if lat is not None else []
            text_val = [f"EM {j + 1}"] if lon is not None else []
            cmds.append(
                f"Plotly.restyle({d(div_id)},"
                f"{{x:{d([x_val])},y:{d([y_val])},text:{d([text_val])}}},{[2*n_em+j]});"
            )

    # ── 4. New RMS history points (extendTraces) ──────────────────────────
    new_rms_y = list(rms_hist[tracker._n_rms_sent:])
    if new_rms_y:
        new_rms_x = list(range(tracker._n_rms_sent, len(rms_hist)))
        cmds.append(
            f"Plotly.extendTraces({d(div_id)},"
            f"{{x:{d([new_rms_x])},y:{d([new_rms_y])}}},{[3*n_em]});"
        )

    # ── 5. Convergence bars (restyle) ─────────────────────────────────────
    patience_pct = min(decomp._n_no_improvement / patience * 100, 100) if patience else 0
    patience_color = "#2ca02c" if patience_pct >= 100 else "#EF553B"
    patience_text = f"{decomp._n_no_improvement:,} / {patience:,}"

    window = list(rms_hist[-decomp.rms_history_window:])
    if len(window) >= 2:
        mean_w = float(np.mean(window))
        var_w = (max(window) - min(window)) / mean_w if mean_w > 0 else float("inf")
        stab_pct = min(decomp.rms_tolerance / max(var_w, 1e-12) * 100, 100)
        stab_color = "#2ca02c" if stab_pct >= 100 else "#00CC96"
        stab_text = f"var={var_w:.4f} / tol={decomp.rms_tolerance:.4f}"
    else:
        stab_pct, stab_color = 0.0, "#00CC96"
        stab_text = f"N/A (need ≥ {decomp.rms_history_window} accepted)"

    cmds.append(
        f"Plotly.restyle({d(div_id)},"
        f"{{x:{d([[patience_pct,stab_pct]])},"
        f"text:{d([[patience_text,stab_text]])},"
        f"'marker.color':{d([[patience_color,stab_color]])}}},{[3*n_em+1]});"
    )

    # ── 6. Title text ─────────────────────────────────────────────────────
    n_visited = len(decomp._visited)
    status = "CONVERGED" if decomp.is_converged else "running"
    best_rms_str = (f"best RMS = {decomp._best_total_rms:.6g}"
                    if decomp._best_total_rms is not None else "")
    main_title = (
        f"EndmemberDecomposition — Live Progress"
        f"  |  combos: {n_visited:,} / {total_combos:.2e}"
        f"  |  {best_rms_str}  |  {status}"
    )
    cmds.append(
        f"Plotly.relayout({d(div_id)},{{'title.text':{d(main_title)}}});"
    )

    # Update tracker's sent-count bookkeeping
    tracker._n_tried_sent = len(decomp._tried_moves)
    tracker._traj_pts_sent = new_traj_pts
    tracker._n_rms_sent = len(rms_hist)

    return "\n".join(cmds)


def create_progress_tracker(decomp) -> ProgressTracker:
    """
    Create a live-updating progress figure for an
    :class:`~specdec.EndmemberDecomposition` run.

    Automatically detects the runtime environment:

    * **Jupyter** — returns a :class:`~plotly.graph_objects.FigureWidget`
      that updates in-place without flickering.
    * **Terminal** — returns a standard :class:`~plotly.graph_objects.Figure`
      written to a temporary HTML file with a 5-second auto-refresh; your
      default browser is opened automatically so you can watch it update.

    Pass the returned :class:`ProgressTracker` to
    :func:`update_progress_tracker` to refresh the display.

    Parameters
    ----------
    decomp : EndmemberDecomposition

    Returns
    -------
    ProgressTracker
    """
    from math import comb as _comb

    n_em = decomp.n_endmembers
    patience = decomp.patience_multiplier * n_em
    total_combos = _comb(decomp._n_candidates, n_em)

    base_fig = _build_progress_figure(n_em, patience, total_combos)
    mode = "jupyter" if _is_jupyter() else "terminal"
    html_path = None
    qt_app = None
    qt_view = None

    # Pre-populate with any existing history (e.g., resumed from checkpoint).
    # Baking data into the HTML avoids a race where extendTraces is injected
    # before Plotly has finished rendering the empty figure after loadFinished.
    if decomp._is_initialized:
        _apply_tracker_updates(base_fig, decomp, n_em, total_combos, patience)

    if mode == "jupyter":
        fig = go.FigureWidget(base_fig)
    else:
        import os, tempfile
        fig = base_fig

        # Reuse the singleton QWebEngineView (creating a second one in the
        # same process causes a segfault in PyQtWebEngine).
        qt_app, qt_view = _get_or_create_qt_view()

        if qt_view is not None:
            import time as _time
            from PyQt5.QtCore import QUrl
            html_path = os.path.join(
                tempfile.gettempdir(),
                f"specdec_progress_pid{os.getpid()}.html",
            )
            html = fig.to_html(
                include_plotlyjs=True, full_html=True,
                div_id=_QT_PROGRESS_DIV_ID,
            )
            # Remove the browser's default body margin so the figure fills
            # the window exactly without triggering scrollbars.
            html = html.replace(
                "<head>",
                "<head><style>"
                "html,body{margin:0;padding:0;overflow:hidden;height:100%;}"
                "</style>",
                1,
            )
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(html)

            qt_view.setWindowTitle("EndmemberDecomposition — Live Progress")
            qt_view.resize(1280, 620)

            # Wait for the page to finish loading (Plotly.js initialises
            # asynchronously; JS injection before loadFinished silently fails).
            _loaded = [False]
            def _on_loaded(ok, _flag=_loaded):
                _flag[0] = True
            qt_view.loadFinished.connect(_on_loaded)
            qt_view.load(QUrl.fromLocalFile(html_path))
            qt_view.show()

            deadline = _time.time() + 15.0
            while not _loaded[0] and _time.time() < deadline:
                qt_app.processEvents()
                _time.sleep(0.05)
            qt_view.loadFinished.disconnect(_on_loaded)
        else:
            # Fall back: self-refreshing HTML in browser
            import webbrowser
            html_path = os.path.join(
                tempfile.gettempdir(),
                f"specdec_progress_pid{os.getpid()}.html",
            )
            _write_progress_html(fig, html_path)
            webbrowser.open(f"file://{html_path}")

    tracker = ProgressTracker(fig, mode, html_path, n_em, total_combos, patience,
                              qt_app=qt_app, qt_view=qt_view)

    # Sync sent-counters with whatever was baked into the figure/HTML so that
    # the first incremental update only sends NEW points, not history.
    if decomp._is_initialized:
        tracker._n_tried_sent = len(decomp._tried_moves)
        tracker._n_rms_sent = len(decomp._accepted_rms_history)
        traj_pts = [0] * n_em
        prev = None
        for step_indices in decomp._em_index_history:
            for j in range(n_em):
                if prev is None or step_indices[j] != prev[j]:
                    if decomp._all_pixels[step_indices[j]].centroid is not None:
                        traj_pts[j] += 1
            prev = step_indices
        tracker._traj_pts_sent = traj_pts

    return tracker


def update_progress_tracker(tracker: ProgressTracker, decomp) -> None:
    """
    Refresh the progress display with the current state of *decomp*.

    In Jupyter mode the ``FigureWidget`` is updated in-place (no flicker).
    In terminal mode the HTML file is rewritten and the browser reloads it
    automatically via its ``<meta http-equiv="refresh">`` tag.

    Parameters
    ----------
    tracker : ProgressTracker
        The object returned by :func:`create_progress_tracker`.
    decomp : EndmemberDecomposition
    """
    fig = tracker.fig

    if tracker.mode == "jupyter":
        with fig.batch_update():
            _apply_tracker_updates(
                fig, decomp, tracker.n_em, tracker.total_combos, tracker.patience
            )
    elif tracker.qt_view is not None:
        # Qt mode: inject only the *new* data points via Plotly.extendTraces /
        # Plotly.restyle — no page reload, so existing points stay visible.
        if tracker.qt_view.isVisible():
            import time as _time
            js = _build_incremental_js(decomp, tracker, _QT_PROGRESS_DIV_ID)
            tracker.qt_view.page().runJavaScript(js)
            for _ in range(5):
                tracker.qt_app.processEvents()
                _time.sleep(0.01)
    else:
        # Browser fallback: rewrite the HTML file
        _apply_tracker_updates(
            fig, decomp, tracker.n_em, tracker.total_combos, tracker.patience
        )
        _write_progress_html(fig, tracker.html_path)
