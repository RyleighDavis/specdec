"""
DecompositionResults — load and explore EndmemberDecomposition results pickles.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .pixel import Pixel


class DecompositionResults:
    """
    Load and explore the output of a completed :class:`EndmemberDecomposition`
    run saved by ``run_decomposition.py``.

    Parameters
    ----------
    path : str or Path
        Path to the results pickle file (e.g. ``ganymede_hst_3em_results.pkl``).

    Attributes
    ----------
    dataset : Dataset
        The full dataset (pixels, spectra, coordinates).
    abundances : ndarray, shape (n_pixels, n_endmembers)
        Per-pixel endmember abundance fractions — always normalised to sum 1.
    scale_factors : ndarray, shape (n_pixels,) or None
        Per-pixel brightness scale factors when the run used ``free_sum=True``;
        ``None`` otherwise.  Full model = ``scale_factors[:, None] * (abundances @ endmember_spectra)``.
    rms_errors : ndarray, shape (n_pixels,)
        Per-pixel RMS residuals of the best fit.
    total_rms : float
        Sum of all per-pixel RMS errors.
    endmembers : list of Pixel
        Best-fit endmember pixel objects.
    endmember_indices : ndarray
        Global dataset indices of the endmember pixels.
    n_iterations, n_accepted : int
        Optimisation iteration counts.
    is_converged : bool
    convergence_reason : str or None
    accepted_rms_history : list of float
        Total RMS at each accepted step.
    em_index_history : list of ndarray
        Endmember index sets at each accepted step.
    n_combinations_tested : int
    params : dict
        Resolved run parameters (see ``run_decomposition.py``).

    Examples
    --------
    ::

        from specdec import DecompositionResults

        r = DecompositionResults("ganymede_hst_3em_results.pkl")
        r.plot_endmember_spectra()
        r.plot_abundance_maps()
        r.plot_pixel("lat+10_lon045")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, path: Union[str, Path]):
        import pickle
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = self._from_dict(data, path=path)
        self.__dict__.update(obj.__dict__)

    @classmethod
    def from_decomposition(cls, decomp) -> "DecompositionResults":
        """
        Construct a :class:`DecompositionResults` directly from an in-memory
        :class:`~specdec.EndmemberDecomposition` instance.

        This is useful for exploring results mid-run or after loading a
        checkpoint, without needing to write and re-read a pickle file::

            decomp = EndmemberDecomposition(ds, n_endmembers=3, free_sum=True)
            decomp.load_checkpoint("run_3em.ckpt")
            r = DecompositionResults.from_decomposition(decomp)
            r.plot_interactive_explorer()

        Parameters
        ----------
        decomp : EndmemberDecomposition

        Returns
        -------
        DecompositionResults
        """
        return cls._from_dict(decomp._build_results_dict(), path=None)

    @classmethod
    def _from_dict(cls, data: dict, path) -> "DecompositionResults":
        """Instantiate from an already-loaded results dict."""
        obj = object.__new__(cls)
        obj.dataset                 = data["dataset"]
        obj.abundances              = data["abundances"]
        obj.scale_factors           = data.get("scale_factors")
        obj.rms_errors              = data["rms_errors"]
        obj.total_rms               = data["total_rms"]
        obj.endmembers              = data["endmembers"]
        obj.endmember_indices       = data["endmember_indices"]
        obj.n_iterations            = data["n_iterations"]
        obj.n_accepted              = data["n_accepted"]
        obj.is_converged            = data["is_converged"]
        obj.convergence_reason      = data["convergence_reason"]
        obj.accepted_rms_history    = data["accepted_rms_history"]
        obj.em_index_history        = data["em_index_history"]
        obj.tried_moves             = data.get("tried_moves", [])
        obj.n_combinations_tested   = data["n_combinations_tested"]
        obj.params                  = data.get("params", {})
        obj._path                   = Path(path) if path is not None else None
        obj._modelled_spectra       = None
        obj._endmember_spectra      = None
        return obj

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def pixels(self) -> List[Pixel]:
        """All pixels in the dataset (same order as ``abundances`` rows)."""
        return self.dataset.pixels

    @property
    def n_pixels(self) -> int:
        return len(self.dataset.pixels)

    @property
    def n_endmembers(self) -> int:
        return self.abundances.shape[1]

    @property
    def wavelengths(self) -> np.ndarray:
        """Wavelength axis from the first endmember."""
        return self.endmembers[0].wavelengths

    @property
    def endmember_spectra(self) -> np.ndarray:
        """Endmember spectra stacked into a matrix, shape (n_em, n_wavelengths)."""
        if self._endmember_spectra is None:
            self._endmember_spectra = np.array(
                [em.spectrum for em in self.endmembers], dtype=float
            )
        return self._endmember_spectra

    @property
    def modelled_spectra(self) -> np.ndarray:
        """
        Full modelled spectra matrix, shape (n_pixels, n_wavelengths).

        Computed as ``abundances @ endmember_spectra`` (times ``scale_factors``
        when ``free_sum=True``).  Result is cached after the first access.
        """
        if self._modelled_spectra is None:
            m = self.abundances @ self.endmember_spectra
            if self.scale_factors is not None:
                m = self.scale_factors[:, None] * m
            self._modelled_spectra = m
        return self._modelled_spectra

    # ------------------------------------------------------------------
    # Plotting — spectra
    # ------------------------------------------------------------------

    def plot_endmember_spectra(self, **kwargs) -> object:
        """
        Plot the best-fit endmember spectra.

        Keyword arguments are forwarded to
        :func:`specdec.plotting.plot_endmember_spectra`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from .plotting import plot_endmember_spectra
        kwargs.setdefault("title", "Endmember Spectra")
        return plot_endmember_spectra(self.endmembers, **kwargs)

    def plot_pixel(
        self,
        pixel: Union[int, str],
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Tuple[object, object]:
        """
        Plot observed vs. modelled spectrum for a single pixel, with
        individual endmember contributions overlaid.

        Parameters
        ----------
        pixel : int or str
            Global pixel index **or** ``pixel_id`` string.
        figsize : (float, float), optional

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        import matplotlib.pyplot as plt
        from .plotting import _COLORS  # used for observed line colour

        if isinstance(pixel, str):
            idx = next(
                (i for i, p in enumerate(self.pixels) if p.pixel_id == pixel),
                None,
            )
            if idx is None:
                raise ValueError(f"No pixel with pixel_id={pixel!r}")
        else:
            idx = int(pixel)

        px = self.pixels[idx]
        dominant = int(np.argmax(self.abundances[idx]))
        obs_color = _COLORS[dominant % len(_COLORS)]

        wl = self.wavelengths
        wl_label = f"Wavelength ({px.wavelength_unit})" if px.wavelength_unit else "Wavelength"
        spec_label = px.spectral_unit or "Relative Reflectance"

        fig, ax = plt.subplots(figsize=figsize or (9, 4))
        ax.plot(wl, px.spectrum, color=obs_color, linewidth=1.5, label="Observed")
        ax.plot(
            wl, self.modelled_spectra[idx],
            color="black", linewidth=1.5, label="Modelled",
        )

        lat = px.metadata.get("lat", "")
        lon = px.metadata.get("lon", "")
        loc = f"  ({lon}°W, {lat}°N)" if lat != "" else ""
        ax.set_title(
            f"{px.pixel_id or f'pixel {idx}'}{loc}"
            f"  |  RMS = {self.rms_errors[idx]:.4g}"
        )
        ax.set_xlabel(wl_label)
        ax.set_ylabel(spec_label)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Plotting — maps
    # ------------------------------------------------------------------

    def plot_abundance_maps(self, **kwargs) -> Tuple[object, np.ndarray]:
        """
        Spatial abundance maps — one panel per endmember.

        Keyword arguments are forwarded to
        :func:`specdec.plotting.plot_abundance_map`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of GeoAxes
        """
        from .plotting import plot_abundance_map
        return plot_abundance_map(
            self.pixels, self.abundances, endmembers=self.endmembers, **kwargs
        )

    def plot_rms_map(self, **kwargs) -> Tuple[object, np.ndarray]:
        """
        Spatial map of per-pixel RMS errors.

        By default uses a ``'hot_r'`` colormap clipped at the 99th-percentile
        RMS value.  Pass ``vmin``, ``vmax``, or ``cmap`` to override.

        Keyword arguments (other than the above) are forwarded to
        :func:`specdec.plotting.plot_abundance_map`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of GeoAxes
        """
        from .plotting import plot_abundance_map
        vmin = kwargs.pop("vmin", 0.0)
        vmax = kwargs.pop("vmax", float(np.percentile(self.rms_errors, 99)))
        cmap = kwargs.pop("cmap", "plasma")
        fig, axes = plot_abundance_map(
            self.pixels,
            self.rms_errors[:, None],
            endmembers=None,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs,
        )
        axes[0].set_title("Per-pixel RMS error")
        return fig, axes

    def plot_residual_map(
        self,
        wavelength: float,
        **kwargs,
    ) -> Tuple[object, np.ndarray]:
        """
        Spatial map of the spectral residual (observed − modelled) at a
        specific wavelength.

        Pixels are coloured by ``observed[wl_idx] - modelled[wl_idx]``, using
        a diverging colormap centred at zero so positive/negative residuals
        are immediately distinguishable.

        Parameters
        ----------
        wavelength : float
            Wavelength at which to evaluate the residual, in the same units
            as the dataset (typically nm).  The nearest available wavelength
            is used.
        vmin, vmax : float, optional
            Colormap limits.  Default: symmetric ±99th-percentile of |residual|.
        cmap : str, optional
            Colormap.  Default ``'RdBu_r'``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of GeoAxes
        """
        from .plotting import plot_abundance_map
        wl_idx = int(np.argmin(np.abs(self.wavelengths - wavelength)))
        actual_wl = float(self.wavelengths[wl_idx])

        obs_col = np.array([p.spectrum[wl_idx] for p in self.pixels])
        residuals = obs_col - self.modelled_spectra[:, wl_idx]

        abs_max = float(np.percentile(np.abs(residuals), 99))
        vmin = kwargs.pop("vmin", -abs_max)
        vmax = kwargs.pop("vmax",  abs_max)
        cmap = kwargs.pop("cmap", "RdBu_r")

        fig, axes = plot_abundance_map(
            self.pixels,
            residuals[:, None],
            endmembers=None,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs,
        )
        wl_unit = self.endmembers[0].wavelength_unit or "nm"
        axes[0].set_title(f"Residual at {actual_wl:.1f} {wl_unit}")
        return fig, axes

    def plot_residual_spectrum(
        self,
        pixel: Union[int, str],
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Tuple[object, object]:
        """
        Plot the spectral residual (observed − modelled) for a single pixel.

        Parameters
        ----------
        pixel : int or str
            Global pixel index or ``pixel_id`` string.

        Returns
        -------
        fig : matplotlib.figure.Figure
        (ax_spec, ax_res) : tuple of Axes
            Top panel (observed + model) and bottom panel (residual).
        """
        import matplotlib.pyplot as plt

        if isinstance(pixel, str):
            idx = next(
                (i for i, p in enumerate(self.pixels) if p.pixel_id == pixel),
                None,
            )
            if idx is None:
                raise ValueError(f"No pixel with pixel_id={pixel!r}")
        else:
            idx = int(pixel)

        px = self.pixels[idx]
        residual = px.spectrum - self.modelled_spectra[idx]
        wl = self.wavelengths
        wl_unit = px.wavelength_unit or "nm"
        spec_unit = px.spectral_unit or "Relative Reflectance"

        color = "#636EFA"
        lat = px.metadata.get("lat", "")
        lon = px.metadata.get("lon", "")
        loc = f"  ({lon}°W, {lat}°N)" if lat != "" else ""

        fig, (ax_spec, ax_res) = plt.subplots(
            2, 1,
            figsize=figsize or (9, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        )

        # Top panel — observed + model
        ax_spec.plot(wl, px.spectrum, color=color, linewidth=1.5, label="Observed")
        ax_spec.plot(wl, self.modelled_spectra[idx], color="black",
                     linewidth=1.5, label="Modelled")
        ax_spec.set_ylabel(spec_unit)
        ax_spec.set_title(
            f"{px.pixel_id or f'pixel {idx}'}{loc}"
            f"  |  RMS = {self.rms_errors[idx]:.4g}"
        )
        ax_spec.legend(fontsize=9)

        # Bottom panel — residual
        ax_res.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_res.plot(wl, residual, color=color, linewidth=1.5)
        ax_res.fill_between(wl, residual, alpha=0.2, color=color)
        ax_res.set_xlabel(f"Wavelength ({wl_unit})")
        ax_res.set_ylabel(f"Residual")

        fig.tight_layout()
        return fig, (ax_spec, ax_res)

    def plot_scale_factor_map(self, **kwargs) -> Tuple[object, np.ndarray]:
        """
        Spatial map of per-pixel scale factors (only meaningful when the run
        used ``free_sum=True``).

        Raises
        ------
        ValueError
            If the run did not use ``free_sum=True``.
        """
        if self.scale_factors is None:
            raise ValueError(
                "scale_factors is None — this run used constrained sum-to-one "
                "(free_sum=False), so there are no scale factors to plot."
            )
        from .plotting import plot_abundance_map
        sf = self.scale_factors
        vmin = kwargs.pop("vmin", float(np.percentile(sf, 1)))
        vmax = kwargs.pop("vmax", float(np.percentile(sf, 99)))
        cmap = kwargs.pop("cmap", "plasma")
        fig, axes = plot_abundance_map(
            self.pixels,
            sf[:, None],
            endmembers=None,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs,
        )
        axes[0].set_title("Per-pixel scale factor")
        return fig, axes

    def plot_abundance_simplex(
        self,
        title: str = "Abundance Simplex",
        show: bool = True,
    ) -> object:
        """
        Plot the per-pixel endmember abundances in the mixing simplex.

        For 3 endmembers a ternary diagram is shown.  For more endmembers a
        2-D barycentric projection is used where N corners sit at equal angles
        and each pixel's position is its abundance-weighted centroid.

        Grey dots show all pixels; coloured paths trace each endmember slot's
        accepted history converging toward its corner; stars mark the final
        best endmembers.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from .plotting import plot_abundance_simplex
        return plot_abundance_simplex(
            abundances=self.abundances,
            endmembers=self.endmembers,
            em_index_history=self.em_index_history,
            endmember_indices=self.endmember_indices,
            pixels=self.pixels,
            title=title,
            show=show,
        )

    def plot_search_history(
        self,
        title: str = "Endmember Search History",
        show: bool = True,
    ) -> object:
        """
        Plot the endmember search trajectory on a lon/lat map.

        Shows every tried position (light scatter) and the accepted trajectory
        (connected dots) for each endmember slot, with the final best positions
        marked as stars.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from .plotting import plot_search_history
        return plot_search_history(
            pixels=self.pixels,
            em_index_history=self.em_index_history,
            endmember_indices=self.endmember_indices,
            tried_moves=self.tried_moves,
            title=title,
            show=show,
        )

    def plot_interactive_explorer(
        self,
        title: str = "Spectral Explorer",
        show: bool = True,
    ) -> object:
        """
        Open an interactive two-panel explorer: a pixel map on the left and
        a spectrum panel on the right.

        Click any pixel on the map to display its observed spectrum, best-fit
        model, and individual endmember contributions in the spectrum panel.
        Endmember pixels are marked with coloured stars.  Each pixel is
        coloured by its dominant endmember.

        Parameters
        ----------
        title : str
            Figure title.
        show : bool
            Open the figure immediately (default ``True``).

        Returns
        -------
        plotly.graph_objects.FigureWidget  (Jupyter)
        plotly.graph_objects.Figure        (terminal)
        """
        from .plotting import plot_interactive_explorer
        return plot_interactive_explorer(
            pixels=self.pixels,
            wavelengths=self.wavelengths,
            abundances=self.abundances,
            endmembers=self.endmembers,
            endmember_indices=self.endmember_indices,
            scale_factors=self.scale_factors,
            rms_errors=self.rms_errors,
            title=title,
            show=show,
        )

    # ------------------------------------------------------------------
    # Plotting — convergence
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Tuple[object, object]:
        """
        Plot total RMS vs. accepted step (convergence history).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize or (7, 3))
        ax.plot(self.accepted_rms_history, color="#636EFA", linewidth=2,
                marker="o", markersize=4)
        ax.set_xlabel("Accepted step")
        ax.set_ylabel("Total RMS")
        ax.set_title(
            f"Convergence — {self.n_iterations:,} iters, "
            f"{self.n_accepted} accepted, "
            f"{self.n_combinations_tested:,} combinations"
        )
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def diagnose(self, verbose: bool = True) -> dict:
        """
        Re-run unmixing on the best endmembers and report solver diagnostics.

        Shows how many pixels used the fast vectorised path, how many needed
        the per-pixel NNLS fallback, and how many of those produced non-finite
        weights and had to fall back further to the clamped unconstrained
        solution.

        Parameters
        ----------
        verbose : bool
            Print a formatted summary.  Default ``True``.

        Returns
        -------
        dict with keys ``n_pixels``, ``n_batch_feasible``, ``n_nnls_fallback``,
        ``n_nonfinite_fallback``.
        """
        from .algorithms import unmix_all
        free_sum = self.params.get("free_sum", False)
        constrain_sum = not free_sum
        non_negative = True

        spectra = np.array([p.spectrum for p in self.pixels], dtype=float)
        diag: dict = {}
        unmix_all(
            spectra,
            self.endmember_spectra,
            constrain_sum=constrain_sum,
            non_negative=non_negative,
            _diagnostics=diag,
        )
        if verbose:
            n = diag.get("n_pixels", self.n_pixels)
            n_fast = diag.get("n_batch_feasible", 0)
            n_nnls = diag.get("n_nnls_fallback", 0)
            n_bad  = diag.get("n_nonfinite_fallback", 0)
            print("=" * 52)
            print("Unmixing diagnostics (best endmembers)")
            print("=" * 52)
            print(f"  Mode               : {'NNLS (free sum)' if free_sum else 'FCLS (sum-to-one)'}")
            print(f"  Total pixels       : {n:,}")
            if "n_batch_feasible" in diag:
                print(f"  Fast batch path    : {n_fast:,}  ({n_fast/n:.1%})")
                print(f"  Per-pixel NNLS     : {n_nnls:,}  ({n_nnls/n:.1%})")
                print(f"  Non-finite fallback: {n_bad:,}  ({n_bad/n:.1%})")
            else:
                print("  (diagnostics only available for NNLS mode)")
            print("=" * 52)
        return diag

    def summary(self) -> None:
        """Print a human-readable summary of the results."""
        conv = "YES" if self.is_converged else "NO"
        print("=" * 55)
        print("DecompositionResults")
        print("=" * 55)
        print(f"  File             : {self._path.name}")
        print(f"  N endmembers     : {self.n_endmembers}")
        print(f"  N pixels         : {self.n_pixels:,}")
        print(f"  Total RMS        : {self.total_rms:.6g}")
        print(f"  Iterations       : {self.n_iterations:,}")
        print(f"  Accepted         : {self.n_accepted}")
        print(f"  Combos tested    : {self.n_combinations_tested:,}")
        print(f"  Converged        : {conv}  ({self.convergence_reason})")
        print(f"  free_sum         : {self.params.get('free_sum', False)}")
        print("-" * 55)
        print("  Best endmembers:")
        for i, em in enumerate(self.endmembers):
            lat = em.metadata.get("lat", "?")
            lon = em.metadata.get("lon", "?")
            print(f"    EM {i + 1}: {em.pixel_id}  ({lon}°W, {lat}°N)")
        print("=" * 55)

    def __repr__(self) -> str:
        conv = "converged" if self.is_converged else "not converged"
        return (
            f"DecompositionResults("
            f"n_endmembers={self.n_endmembers}, "
            f"n_pixels={self.n_pixels:,}, "
            f"total_rms={self.total_rms:.6g}, "
            f"status={conv!r})"
        )
