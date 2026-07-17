"""
EndmemberDecomposition — iterative endmember selection and spectral unmixing.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from math import comb
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from joblib import Parallel, delayed

from .pixel import Pixel
from .dataset import Dataset
from .algorithms import (
    initialize_endmembers_kmeans,
    unmix_all,
    _evaluate_combination,
    smooth_endmember_spectra,
)
from . import config
from .plotting import plot_endmember_spectra


# Mutable attributes persisted in a checkpoint (order does not matter).
_CHECKPOINT_FIELDS: Tuple[str, ...] = (
    "_current_em_indices",
    "_current_abundances",
    "_current_scale_factors",
    "_current_rms_errors",
    "_current_total_rms",
    "_best_em_indices",
    "_best_abundances",
    "_best_scale_factors",
    "_best_rms_errors",
    "_best_total_rms",
    "_visited",
    "_accepted_rms_history",
    "_em_index_history",
    "_tried_moves",
    "_n_iterations",
    "_n_accepted",
    "_n_no_improvement",
    "_n_positions_exhausted_in_a_row",
    "_stuck",
    "_is_initialized",
    "_is_converged",
    "_convergence_reason",
)


class EndmemberDecomposition:
    """
    Iterative endmember selection and linear spectral unmixing.

    The algorithm
    -------------
    1. **Initialise** *N* endmember pixels via K-means + SAM (or user-supplied
       indices).
    2. **Decompose** every pixel as a linear combination of the current
       endmembers; record the total summed RMS error.
    3. **Perturb**: rotating through endmembers 0 … N-1, randomly pick a
       replacement pixel from those that carry ≥ *endmember_threshold* fraction
       of that endmember's abundance (matching the paper's method).
    4. **Accept** the change if the new total RMS is strictly lower.
    5. **Broaden** if stuck: once every single-position swap of the current
       state has already been tried and rejected, proposals widen to
       changing 2 or more endmember positions at once instead (still under
       the same accept-only-if-better rule in step 4) so the search isn't
       permanently confined to one small neighbourhood. See
       :meth:`_sample_broadened_combo`.
    6. **Repeat** steps 3–5 until a convergence condition is met.

    Parallel mode (``n_jobs > 1``)
    --------------------------------
    When ``n_jobs > 1``, each perturbation step evaluates up to *n_jobs*
    candidate replacements in parallel (using :mod:`joblib`'s ``loky``
    process-based backend) instead of just one, then accepts the single
    best-improving candidate.  This gives an approximately *O(n_jobs)* speedup
    in combinations evaluated per unit of wall time, and therefore in
    convergence speed.

    .. note::
        Parallel evaluation uses threads (not processes), so the full spectra
        matrix is shared rather than copied — keeping memory usage flat
        regardless of worker count.

    Convergence conditions (evaluated after each step, in order)
    ------------------------------------------------------------
    1. **Hard stop** — every possible combination of *N* endmembers from the
       candidate set has been evaluated. In practice this essentially never
       happens on a real dataset (``C(n_candidates, n_endmembers)`` is
       typically in the millions), but it's checked because it's a genuine
       proof of global optimality when it does.
    2. **Custom** — a user-supplied ``convergence_fn(decomp) -> bool`` returns
       ``True``.

    There is no other automatic stopping heuristic. This is a plain greedy
    hill-climb: a step is only ever accepted if it strictly improves the
    *current* total RMS, so once every reachable neighbour of the current
    state has already been visited, further iterations stop finding anything
    new to accept and ``current_total_rms`` stops changing -- but the run
    itself keeps going rather than declaring victory, since a plateau in one
    local neighbourhood says nothing about whether it's the best one. Use
    *max_iterations* or interrupt the run (``Ctrl+C``, or the Qt stop button
    under ``show_progress=True``) to stop it deliberately; either way the
    best result found so far is preserved.

    Parameters
    ----------
    dataset : Dataset
        The dataset.  Candidate status of pixels is snapshotted at
        construction time.
    n_endmembers : int
        Number of endmember pixels to identify.
    initial_endmembers : list of int or list of Pixel, optional
        Explicit starting endmembers (by global dataset index or Pixel
        object).  Length must equal *n_endmembers*.  If ``None``, the
        K-means + SAM heuristic is used.
    constrain_sum : bool
        Enforce abundance sum-to-one constraint.  Default ``True``.
    non_negative : bool
        Enforce non-negative abundances.  Default ``True``.
    endmember_threshold : float, optional
        Minimum abundance fraction for a pixel to be considered when
        perturbing a given endmember.  Default
        :attr:`DEFAULT_ENDMEMBER_THRESHOLD` (0.50).
    free_sum : bool, optional
        When ``True``, the per-pixel abundances are **not** forced to sum to
        one during the fit.  Instead, non-negative least squares (NNLS) is
        used, and the raw weights are normalised to unit sum before storage.
        The ``scale_factors`` attribute then holds the per-pixel sum of the
        raw weights, which acts as a brightness/albedo scaling term:

            modelled spectrum ≈ scale_factor × (normalised abundances @ endmembers)

        RMS is always computed against the unscaled model (raw weights @
        endmembers), so the endmember search optimises the true spectral fit.
        Default ``False`` (fully constrained sum-to-one, i.e. FCLS).
    n_jobs : int or float, optional
        Number of parallel workers used for batch candidate evaluation:

        * ``1`` (default) — fully sequential; no parallelism overhead.
        * Positive ``int`` — use exactly that many workers.
        * ``-1`` — use all logical CPUs (``os.cpu_count()``).
        * ``float`` in ``(0.0, 1.0]`` — use that fraction of logical CPUs,
          rounded to the nearest integer (minimum 1).

        Each step evaluates up to *n_jobs* candidate combinations in parallel
        and accepts whichever one gives the greatest RMS improvement (if any).
    minimization_fn : callable, optional
        Custom per-pixel solver replacing the default NNLS / SLSQP routine.
        Signature::

            fn(spectrum, endmember_spectra) -> (abundances, rms)

        When ``n_jobs > 1`` this callable must be thread-safe.
    convergence_fn : callable, optional
        Custom convergence predicate.  Signature::

            fn(decomp: EndmemberDecomposition) -> bool

        Return ``True`` to signal convergence.  Called *after* the hard-stop
        check.
    random_state : int, optional
        Seed for the internal NumPy RNG (used in perturbation sampling and
        K-means initialisation).
    smooth_endmembers : bool
        Apply a Savitzky-Golay filter to each candidate endmember's
        spectrum before it's used as a basis vector in the unmixing fit
        (both during the search and for the final accepted endmembers).
        Endmembers are real observed pixels, so they carry the same
        per-wavelength noise as any other pixel -- but because a given
        endmember is reused across every pixel it contributes to, that
        noise gets replicated (scaled by abundance) into every modelled
        spectrum rather than staying local to one pixel. Smoothing only
        the endmembers (never the observed target spectra being fit)
        reduces that. Default ``True``.
    endmember_smoothing_window : int
        Savitzky-Golay window length in pixels; must be odd and greater
        than *endmember_smoothing_polyorder*. Only used when
        *smooth_endmembers* is ``True``. Default ``7``.
    endmember_smoothing_polyorder : int
        Polynomial order fit within each smoothing window. Default ``2``
        (quadratic) -- preserves absorption band shape/depth reasonably
        well at typical window widths while still suppressing
        single-pixel noise spikes; only used when *smooth_endmembers* is
        ``True``.

    Attributes
    ----------
    DEFAULT_ENDMEMBER_THRESHOLD : float
        Class-level default for *endmember_threshold*.
    """

    # ------------------------------------------------------------------
    # Class-level defaults (easily patched at module level if desired)
    # ------------------------------------------------------------------
    DEFAULT_ENDMEMBER_THRESHOLD: float = 0.50

    def __init__(
        self,
        dataset: Dataset,
        n_endmembers: int,
        initial_endmembers: Optional[List] = None,
        constrain_sum: bool = True,
        non_negative: bool = True,
        free_sum: bool = False,
        endmember_threshold: Optional[float] = None,
        n_jobs: Union[int, float] = 1,
        minimization_fn: Optional[Callable] = None,
        convergence_fn: Optional[Callable] = None,
        random_state: Optional[int] = None,
        smooth_endmembers: bool = True,
        endmember_smoothing_window: int = 7,
        endmember_smoothing_polyorder: int = 2,
    ):
        # ---- validate n_endmembers ----
        n_candidates = len(dataset.candidate_pixels)
        if n_endmembers < 2:
            raise ValueError("n_endmembers must be at least 2.")
        if n_endmembers > n_candidates:
            raise ValueError(
                f"n_endmembers ({n_endmembers}) exceeds the number of "
                f"candidate pixels ({n_candidates})."
            )

        # ---- validate endmember smoothing ----
        # Fail fast at construction rather than on the first accepted step --
        # smooth_endmember_spectra() would raise the same errors lazily.
        if smooth_endmembers:
            if endmember_smoothing_window % 2 == 0:
                raise ValueError(
                    f"endmember_smoothing_window must be odd, got "
                    f"{endmember_smoothing_window}."
                )
            if endmember_smoothing_window <= endmember_smoothing_polyorder:
                raise ValueError(
                    f"endmember_smoothing_window ({endmember_smoothing_window}) "
                    f"must be greater than endmember_smoothing_polyorder "
                    f"({endmember_smoothing_polyorder})."
                )

        self.smooth_endmembers = smooth_endmembers
        self.endmember_smoothing_window = endmember_smoothing_window
        self.endmember_smoothing_polyorder = endmember_smoothing_polyorder

        self.dataset = dataset
        self.n_endmembers = n_endmembers
        self.free_sum = free_sum
        # free_sum uses NNLS (no sum constraint, non-negative); post-normalises
        # the raw weights to unit sum before storing them as abundances.
        self.constrain_sum = False if free_sum else constrain_sum
        self.non_negative = True if free_sum else non_negative
        self.minimization_fn = minimization_fn
        self.convergence_fn = convergence_fn
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._n_workers: int = self._resolve_n_jobs(n_jobs)
        # macOS: Apple's Accelerate BLAS deadlocks when called from multiple
        # Python threads simultaneously.  Use loky (spawn-based processes) there
        # instead — spawn starts workers clean so there's no fork-memory blowup.
        # Linux: threading shares memory with the parent, keeping footprint flat.
        self._parallel_backend: str = "loky" if sys.platform == "darwin" else "threading"

        # ---- per-instance tuneable parameters ----
        self.endmember_threshold: float = (
            endmember_threshold
            if endmember_threshold is not None
            else self.DEFAULT_ENDMEMBER_THRESHOLD
        )
        # ---- snapshot pixel lists and build indices ----
        self._all_pixels: List[Pixel] = dataset.pixels
        self._candidate_pixels: List[Pixel] = dataset.candidate_pixels
        self._n_pixels: int = len(self._all_pixels)
        self._n_candidates: int = len(self._candidate_pixels)

        # pixel object → index into _all_pixels
        self._pixel_to_idx: Dict[Pixel, int] = {
            p: i for i, p in enumerate(self._all_pixels)
        }
        # indices of candidate pixels in _all_pixels
        self._candidate_indices: np.ndarray = np.array(
            [self._pixel_to_idx[p] for p in self._candidate_pixels], dtype=int
        )
        # set of candidate indices for O(1) lookup
        self._candidate_index_set: set = set(self._candidate_indices.tolist())

        # ---- spectra matrices ----
        self._all_spectra: np.ndarray = dataset.get_spectra_matrix(self._all_pixels)
        self._candidate_spectra: np.ndarray = dataset.get_spectra_matrix(
            self._candidate_pixels
        )

        # ---- resolve user-supplied initial endmembers ----
        self._user_initial_indices: Optional[np.ndarray] = (
            self._resolve_initial_endmembers(initial_endmembers)
            if initial_endmembers is not None
            else None
        )

        # ---- mutable optimisation state ----
        self._current_em_indices: Optional[np.ndarray] = None
        self._current_abundances: Optional[np.ndarray] = None
        self._current_scale_factors: Optional[np.ndarray] = None  # free_sum only
        self._current_rms_errors: Optional[np.ndarray] = None
        self._current_total_rms: Optional[float] = None

        self._best_em_indices: Optional[np.ndarray] = None
        self._best_abundances: Optional[np.ndarray] = None
        self._best_scale_factors: Optional[np.ndarray] = None     # free_sum only
        self._best_rms_errors: Optional[np.ndarray] = None
        self._best_total_rms: Optional[float] = None

        # combination key → total RMS
        self._visited: Dict[frozenset, float] = {}
        # total RMS at each *accepted* step (including initialisation)
        self._accepted_rms_history: List[float] = []
        # endmember global-index arrays at each accepted step (including init)
        self._em_index_history: List[np.ndarray] = []
        # (em_position, pixel_global_idx) for every candidate tried
        self._tried_moves: List[Tuple[int, int]] = []

        self._n_iterations: int = 0
        self._n_accepted: int = 0
        self._n_no_improvement: int = 0
        # Consecutive positions (in the em_position rotation) for which every
        # reachable single-swap neighbour of the *current* state was already
        # visited -- once this reaches n_endmembers, _stuck is set (see
        # step() / _sample_broadened_combo()).
        self._n_positions_exhausted_in_a_row: int = 0
        # Whether every single-position-swap neighbour of the current state
        # has already been visited, so step() should propose a broadened
        # (multi-position) move instead. Cleared the moment any move (swap
        # or broadened) is accepted, since that changes the current state
        # and its single-swap neighbourhood is generally unexplored again.
        self._stuck: bool = False
        self._is_initialized: bool = False
        self._is_converged: bool = False
        self._convergence_reason: Optional[str] = None
        self._stop_requested: bool = False  # set by main thread on Ctrl+C with show_progress

        # Cached NNLS solver diagnostics (populated by _update_diagnostics)
        self._last_diag: Dict = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_diagnostics(self) -> None:
        """
        Re-run unmixing on the current best endmembers to refresh ``_last_diag``.

        Called whenever a new best solution is found (and once at initialisation)
        so the solver breakdown is always up-to-date without per-iteration
        overhead.  No-op when ``free_sum=False`` (NNLS diagnostics are not
        relevant for FCLS).
        """
        if not self.free_sum or self._best_em_indices is None:
            return
        unmix_all(
            self._all_spectra,
            self._em_spectra(self._best_em_indices),
            constrain_sum=self.constrain_sum,
            non_negative=self.non_negative,
            minimization_fn=self.minimization_fn,
            _diagnostics=self._last_diag,
        )

    def _apply_normalization(
        self, weights: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        When ``free_sum=True``, normalise raw NNLS weights to unit sum and
        return the per-pixel scale factors.  Otherwise return weights unchanged
        with ``None`` for scale factors.

        Parameters
        ----------
        weights : ndarray, shape (n_pixels, n_endmembers)

        Returns
        -------
        abundances : ndarray, shape (n_pixels, n_endmembers)
            Normalised (sum-to-one) abundances.
        scale_factors : ndarray, shape (n_pixels,) or None
            Per-pixel sum of raw weights.  ``None`` when ``free_sum=False``.
        """
        if not self.free_sum:
            return weights, None
        scale_factors = weights.sum(axis=1)                    # (n_pixels,)
        safe_denom = np.maximum(scale_factors, 1e-12)
        abundances = weights / safe_denom[:, None]
        return abundances, scale_factors

    @staticmethod
    def _resolve_n_jobs(n_jobs: Union[int, float]) -> int:
        """Convert the *n_jobs* argument to a concrete integer worker count."""
        n_cpus = os.cpu_count() or 1
        if isinstance(n_jobs, float):
            if not (0.0 < n_jobs <= 1.0):
                raise ValueError(
                    f"float n_jobs must be in (0.0, 1.0], got {n_jobs}."
                )
            return max(1, round(n_jobs * n_cpus))
        if isinstance(n_jobs, int):
            if n_jobs == -1:
                return n_cpus
            if n_jobs < 1:
                raise ValueError(
                    f"int n_jobs must be >= 1 or -1 (use all CPUs), got {n_jobs}."
                )
            return n_jobs
        raise TypeError(
            f"n_jobs must be an int or float in (0, 1], got {type(n_jobs).__name__}."
        )

    def _resolve_initial_endmembers(self, initial_endmembers) -> np.ndarray:
        """Convert user-supplied initial endmembers to global indices."""
        indices: List[int] = []
        for em in initial_endmembers:
            if isinstance(em, Pixel):
                if em not in self._pixel_to_idx:
                    raise ValueError(f"Pixel {em!r} not found in dataset.")
                indices.append(self._pixel_to_idx[em])
            elif isinstance(em, (int, np.integer)):
                idx = int(em)
                if not (0 <= idx < self._n_pixels):
                    raise IndexError(
                        f"Index {idx} is out of range [0, {self._n_pixels})."
                    )
                indices.append(idx)
            else:
                raise TypeError(
                    "initial_endmembers must contain int indices or Pixel objects; "
                    f"got {type(em).__name__}."
                )

        if len(indices) != self.n_endmembers:
            raise ValueError(
                f"Expected {self.n_endmembers} initial endmembers, "
                f"got {len(indices)}."
            )
        return np.array(indices, dtype=int)

    @staticmethod
    def _make_key(indices: np.ndarray) -> frozenset:
        return frozenset(int(i) for i in indices)

    def _em_spectra(self, indices: np.ndarray) -> np.ndarray:
        spectra = self._all_spectra[indices]
        if self.smooth_endmembers:
            spectra = smooth_endmember_spectra(
                spectra,
                self.endmember_smoothing_window,
                self.endmember_smoothing_polyorder,
            )
        return spectra

    def _compute_models(
        self, em_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run unmix_all for the given endmember indices."""
        return unmix_all(
            self._all_spectra,
            self._em_spectra(em_indices),
            constrain_sum=self.constrain_sum,
            non_negative=self.non_negative,
            minimization_fn=self.minimization_fn,
        )

    def _get_perturbation_candidates(
        self, em_position: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Return global pixel indices eligible to replace the endmember at
        position *em_position*, plus whether any of them lead to a
        not-yet-visited combination.

        Eligible pixels must:
        * be in the candidate set,
        * have abundance ≥ *endmember_threshold* for endmember *em_position*,
        * not already be one of the current endmembers.

        Falls back to all candidate pixels (minus current endmembers) if no
        pixels satisfy the abundance criterion.

        Candidates whose resulting combination (the current endmembers with
        *em_position* swapped for that pixel) is already in :attr:`_visited`
        are excluded from the returned array whenever at least one
        not-yet-visited candidate remains. Proposing an already-visited
        combination again can never be accepted in a search that only ever
        accepts strict improvements: it was already evaluated against a
        *current* RMS that (without ever regressing) can only have been ≥ the
        current RMS now, so if it wasn't an improvement then, it can't be one
        now either -- it's a guaranteed-useless iteration, wasting a step
        that could have tested something new.

        If every eligible candidate has already been tried against the
        current co-endmembers, the fully-visited eligible set is returned
        as-is (so the caller always has something non-empty to sample from
        when *eligible* is non-empty) along with ``any_unvisited=False`` --
        this signals that single-position swaps at *this* position can't
        reach anything new right now. Once every position agrees (tracked in
        :meth:`step` via ``_n_positions_exhausted_in_a_row``), the search
        marks itself :attr:`_stuck` and proposals broaden to 2+ positions at
        once (see :meth:`_sample_broadened_combo`) instead of continuing to
        re-hit this same fully-visited set.
        """
        em_abundances = self._current_abundances[:, em_position]  # (n_pixels,)
        current_em_set = set(self._current_em_indices.tolist())

        # candidates with high enough abundance
        high_mask = em_abundances >= self.endmember_threshold
        eligible = np.array(
            [
                i
                for i in self._candidate_indices
                if high_mask[i] and i not in current_em_set
            ],
            dtype=int,
        )

        if len(eligible) == 0:
            # Fallback: any candidate that is not a current endmember
            eligible = np.array(
                [i for i in self._candidate_indices if i not in current_em_set],
                dtype=int,
            )

        any_unvisited = False
        if len(eligible) > 0:
            base_key = frozenset(
                int(idx)
                for pos, idx in enumerate(self._current_em_indices)
                if pos != em_position
            )
            unvisited = np.array(
                [i for i in eligible if (base_key | {int(i)}) not in self._visited],
                dtype=int,
            )
            any_unvisited = len(unvisited) > 0
            if any_unvisited:
                eligible = unvisited

        return eligible, any_unvisited

    def _accept(
        self,
        new_indices: np.ndarray,
        weights: np.ndarray,
        rms_errors: np.ndarray,
        total_rms: float,
    ):
        """Record an accepted perturbation and update best state if improved."""
        abundances, scale_factors = self._apply_normalization(weights)

        self._current_em_indices = new_indices.copy()
        self._current_abundances = abundances
        self._current_scale_factors = scale_factors
        self._current_rms_errors = rms_errors
        self._current_total_rms = total_rms
        self._n_accepted += 1
        self._accepted_rms_history.append(total_rms)
        self._em_index_history.append(new_indices.copy())

        if total_rms < self._best_total_rms:
            self._best_em_indices = new_indices.copy()
            self._best_abundances = abundances.copy()
            self._best_scale_factors = (
                scale_factors.copy() if scale_factors is not None else None
            )
            self._best_rms_errors = rms_errors.copy()
            self._best_total_rms = total_rms
            self._update_diagnostics()

    def _check_convergence(self) -> Tuple[bool, Optional[str]]:
        """Return (converged, reason_string)."""

        # 1. Hard stop: all combinations have been tried
        total_combos = comb(self._n_candidates, self.n_endmembers)
        if len(self._visited) >= total_combos:
            return True, "exhausted_all_combinations"

        # 2. Custom convergence predicate
        if self.convergence_fn is not None and self.convergence_fn(self):
            return True, "custom_convergence_fn"

        return False, None

    # ------------------------------------------------------------------
    # Step internals — sequential and parallel paths
    # ------------------------------------------------------------------

    def _try_one_candidate(self, em_position: int, candidates: np.ndarray) -> bool:
        """
        Sequential path: draw one random candidate, evaluate, accept if better.
        """
        new_pixel_idx = int(self._rng.choice(candidates))
        self._tried_moves.append((em_position, new_pixel_idx))
        new_indices = self._current_em_indices.copy()
        new_indices[em_position] = new_pixel_idx
        key = self._make_key(new_indices)

        if key in self._visited:
            new_total_rms = self._visited[key]
            if new_total_rms < self._current_total_rms:
                abundances, rms_errors, new_total_rms = self._compute_models(new_indices)
                self._accept(new_indices, abundances, rms_errors, new_total_rms)
                return True
        else:
            abundances, rms_errors, new_total_rms = self._compute_models(new_indices)
            self._visited[key] = new_total_rms
            if new_total_rms < self._current_total_rms:
                self._accept(new_indices, abundances, rms_errors, new_total_rms)
                return True
        return False

    def _try_candidates_parallel(
        self, em_position: int, candidates: np.ndarray
    ) -> bool:
        """
        Parallel path: draw up to *n_workers* candidates, evaluate them all in
        parallel, then accept whichever gives the greatest improvement (if any).

        Combinations already in the visited cache skip re-evaluation.  Any
        newly evaluated combination is stored in the cache regardless of
        whether it is accepted.
        """
        n_to_try = min(self._n_workers, len(candidates))
        sampled = self._rng.choice(candidates, size=n_to_try, replace=False)

        # Partition into cached (RMS already known) vs needs evaluation
        cached: List[Tuple[np.ndarray, float]] = []
        to_eval: List[Tuple[np.ndarray, frozenset]] = []

        for new_pixel_idx in sampled:
            self._tried_moves.append((em_position, int(new_pixel_idx)))
            new_indices = self._current_em_indices.copy()
            new_indices[em_position] = int(new_pixel_idx)
            key = self._make_key(new_indices)
            if key in self._visited:
                cached.append((new_indices, self._visited[key]))
            else:
                to_eval.append((new_indices, key))

        # Evaluate all uncached combinations in parallel
        eval_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        if to_eval:
            outputs = Parallel(n_jobs=self._n_workers, backend=self._parallel_backend)(
                delayed(_evaluate_combination)(
                    self._all_spectra,
                    indices,
                    self.constrain_sum,
                    self.non_negative,
                    self.minimization_fn,
                    self.endmember_smoothing_window if self.smooth_endmembers else None,
                    self.endmember_smoothing_polyorder,
                )
                for indices, _ in to_eval
            )
            for (indices, key), (abundances, rms_errors, total_rms) in zip(
                to_eval, outputs
            ):
                self._visited[key] = total_rms
                eval_results.append((indices, abundances, rms_errors, total_rms))

        # Find the single best improvement across all evaluated candidates
        best_rms = self._current_total_rms
        best_indices: Optional[np.ndarray] = None
        best_abundances: Optional[np.ndarray] = None
        best_rms_errors: Optional[np.ndarray] = None

        # Check cached results (abundances not stored — will recompute on accept)
        for new_indices, total_rms in cached:
            if total_rms < best_rms:
                best_rms = total_rms
                best_indices = new_indices
                best_abundances = None
                best_rms_errors = None

        # Check freshly evaluated results (abundances already available)
        for indices, abundances, rms_errors, total_rms in eval_results:
            if total_rms < best_rms:
                best_rms = total_rms
                best_indices = indices
                best_abundances = abundances
                best_rms_errors = rms_errors

        if best_indices is not None:
            if best_abundances is None:
                # Winner came from cache — recompute full models for state update
                best_abundances, best_rms_errors, best_rms = self._compute_models(
                    best_indices
                )
            self._accept(best_indices, best_abundances, best_rms_errors, best_rms)
            return True

        return False

    def _sample_broadened_combo(self) -> Optional[np.ndarray]:
        """
        Sample a not-yet-visited endmember combination that differs from the
        current one in 2 or more positions at once (chosen at random), for
        use once single-position swaps are exhausted (see :meth:`step`).

        A single-swap move can only ever reach the up to
        ``n_endmembers * (n_candidates - n_endmembers)`` combinations that
        differ from the current one in exactly one position -- once those
        are all visited, the current state is fixed forever under swap-only
        moves and combinations outside that neighbourhood are permanently
        unreachable. Changing 2+ positions at once escapes that ceiling
        while keeping the same accept-only-if-strictly-better rule as every
        other proposal -- there's no separate "restart" event, this is just
        a wider version of the same move.

        Returns
        -------
        ndarray or None
            A candidate global-index array, or ``None`` if no not-yet-visited
            combination could be found within the attempt budget (i.e. the
            search is at or very near exhausting the full combination space
            -- :meth:`_check_convergence`'s hard stop handles that case).
        """
        max_attempts = 200
        n = self.n_endmembers
        current = self._current_em_indices
        for _ in range(max_attempts):
            k = int(self._rng.integers(2, n + 1))  # 2 .. n inclusive
            positions = self._rng.choice(n, size=k, replace=False)
            kept = {int(idx) for pos, idx in enumerate(current) if pos not in positions}
            pool = np.array(
                [i for i in self._candidate_indices if i not in kept], dtype=int
            )
            if len(pool) < k:
                continue
            chosen = self._rng.choice(pool, size=k, replace=False)
            new_indices = current.copy()
            for pos, val in zip(positions, chosen):
                new_indices[pos] = int(val)
            key = self._make_key(new_indices)
            if key not in self._visited:
                return new_indices
        return None

    def _try_broadened_candidate(self) -> bool:
        """Sequential path for a broadened (2+ position) proposal."""
        new_indices = self._sample_broadened_combo()
        if new_indices is None:
            return False
        abundances, rms_errors, total_rms = self._compute_models(new_indices)
        self._visited[self._make_key(new_indices)] = total_rms
        if total_rms < self._current_total_rms:
            self._accept(new_indices, abundances, rms_errors, total_rms)
            return True
        return False

    def _try_broadened_candidates_parallel(self) -> bool:
        """Parallel path: sample up to *n_workers* broadened proposals, evaluate
        them all in parallel, then accept whichever gives the greatest
        improvement (if any)."""
        proposals: List[Tuple[np.ndarray, frozenset]] = []
        seen_keys: set = set()
        for _ in range(self._n_workers):
            new_indices = self._sample_broadened_combo()
            if new_indices is None:
                break
            key = self._make_key(new_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            proposals.append((new_indices, key))

        if not proposals:
            return False

        outputs = Parallel(n_jobs=self._n_workers, backend=self._parallel_backend)(
            delayed(_evaluate_combination)(
                self._all_spectra,
                indices,
                self.constrain_sum,
                self.non_negative,
                self.minimization_fn,
                self.endmember_smoothing_window if self.smooth_endmembers else None,
                self.endmember_smoothing_polyorder,
            )
            for indices, _ in proposals
        )

        best_rms = self._current_total_rms
        best_indices: Optional[np.ndarray] = None
        best_abundances: Optional[np.ndarray] = None
        best_rms_errors: Optional[np.ndarray] = None
        for (indices, key), (abundances, rms_errors, total_rms) in zip(proposals, outputs):
            self._visited[key] = total_rms
            if total_rms < best_rms:
                best_rms = total_rms
                best_indices = indices
                best_abundances = abundances
                best_rms_errors = rms_errors

        if best_indices is not None:
            self._accept(best_indices, best_abundances, best_rms_errors, best_rms)
            return True
        return False

    # ------------------------------------------------------------------
    # Public API — initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> "EndmemberDecomposition":
        """
        Initialise the endmember set.

        Uses K-means + SAM if no *initial_endmembers* were given,
        otherwise uses the user-specified pixels.

        Returns
        -------
        self
        """
        if self._user_initial_indices is not None:
            init_indices = self._user_initial_indices.copy()
            self._kmeans_centers: Optional[np.ndarray] = None
        else:
            init_indices, self._kmeans_centers = initialize_endmembers_kmeans(
                self._candidate_spectra,
                self.n_endmembers,
                pixel_indices=self._candidate_indices,
                random_state=self.random_state,
            )

        # Compute initial models
        weights, rms_errors, total_rms = self._compute_models(init_indices)
        abundances, scale_factors = self._apply_normalization(weights)
        key = self._make_key(init_indices)
        self._visited[key] = total_rms

        self._current_em_indices = init_indices.copy()
        self._current_abundances = abundances
        self._current_scale_factors = scale_factors
        self._current_rms_errors = rms_errors
        self._current_total_rms = total_rms

        self._best_em_indices = init_indices.copy()
        self._best_abundances = abundances.copy()
        self._best_scale_factors = (
            scale_factors.copy() if scale_factors is not None else None
        )
        self._best_rms_errors = rms_errors.copy()
        self._best_total_rms = total_rms

        self._accepted_rms_history.append(total_rms)
        self._em_index_history.append(init_indices.copy())
        self._is_initialized = True
        self._update_diagnostics()

        if config.show_plots:
            plot_endmember_spectra(
                self.endmembers,
                cluster_centers=self._kmeans_centers,
                title="Initial Endmember Spectra",
            )

        return self

    # ------------------------------------------------------------------
    # Public API — single step
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Perform one perturbation–evaluation–accept/reject step.

        Normally this is a single-position swap: rotating through endmember
        positions 0 … N-1, one candidate replacement is proposed for the
        current position (or up to *n_jobs* in parallel mode) and accepted
        if it strictly improves the total RMS -- matching the paper's
        method.  Once every reachable single-swap neighbour of the current
        state has already been visited (:attr:`_stuck`), single-position
        swaps can't reach anything new -- see :meth:`_sample_broadened_combo`
        -- so proposals broaden to changing 2 or more positions at once
        instead, under the exact same accept-only-if-better rule, until an
        improvement is found (which resumes single-swap proposals from the
        new state) or the full combination space is exhausted.

        Returns
        -------
        bool
            ``True`` if the perturbation was accepted (total RMS improved).

        Raises
        ------
        RuntimeError
            If :meth:`initialize` has not been called yet.
        """
        if not self._is_initialized:
            raise RuntimeError("Call initialize() before step().")
        if self._is_converged:
            return False

        if not self._stuck:
            em_position = self._n_iterations % self.n_endmembers
            candidates, any_unvisited = self._get_perturbation_candidates(em_position)

            accepted = False
            if len(candidates) > 0:
                if self._n_workers == 1:
                    accepted = self._try_one_candidate(em_position, candidates)
                else:
                    accepted = self._try_candidates_parallel(em_position, candidates)

            if accepted:
                self._n_positions_exhausted_in_a_row = 0
            elif any_unvisited:
                self._n_positions_exhausted_in_a_row = 0
            else:
                self._n_positions_exhausted_in_a_row += 1
                if self._n_positions_exhausted_in_a_row >= self.n_endmembers:
                    self._stuck = True
        else:
            if self._n_workers == 1:
                accepted = self._try_broadened_candidate()
            else:
                accepted = self._try_broadened_candidates_parallel()

        self._n_iterations += 1
        if accepted:
            self._n_no_improvement = 0
            self._n_positions_exhausted_in_a_row = 0
            self._stuck = False
        else:
            self._n_no_improvement += 1

        converged, reason = self._check_convergence()
        if converged:
            self._is_converged = True
            self._convergence_reason = reason

        return accepted

    # ------------------------------------------------------------------
    # Public API — full run
    # ------------------------------------------------------------------

    def run(
        self,
        max_iterations: Optional[int] = None,
        verbose: bool = True,
        progress_interval: int = 100,
        show_progress: bool = False,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 500,
        results_path: Optional[str] = None,
        _progress_tracker=None,
    ) -> "EndmemberDecomposition":
        """
        Run the full iterative optimisation until convergence.

        Parameters
        ----------
        max_iterations : int, optional
            Hard cap on the number of iterations.  If ``None``, runs until
            one of the convergence conditions is satisfied.
        verbose : bool
            Print progress messages.  Default ``True``.
        progress_interval : int
            Print a progress line (and update the live plot if
            ``show_progress=True``) every *n* iterations.  Default 100.
        show_progress : bool
            Display a live-updating Plotly progress figure in a Jupyter
            notebook.  Requires ``plotly`` and ``IPython``.  Default ``False``.
        checkpoint_path : str or Path, optional
            If given, the optimisation state is saved to this file every
            *checkpoint_interval* iterations and whenever the run is
            interrupted with ``KeyboardInterrupt``.  Pass the same path to
            :meth:`load_checkpoint` on a new instance to resume.
        checkpoint_interval : int
            Save a checkpoint every this many iterations.  Default 500.
        results_path : str or Path, optional
            If given, a results pickle is written to this path on completion
            (and on ``KeyboardInterrupt`` so partial results are not lost).
            The file can be loaded directly by
            :class:`~specdec.DecompositionResults`.

        Returns
        -------
        self
        """
        self._max_iterations = max_iterations  # stored so save_results can include it
        if not self._is_initialized:
            self.initialize()
        elif self.free_sum and not self._last_diag:
            # Resuming from checkpoint — populate diagnostics before first print
            self._update_diagnostics()

        total_combos = comb(self._n_candidates, self.n_endmembers)

        if verbose:
            print("=" * 60)
            print("specdec: Endmember Optimisation")
            print("=" * 60)
            print(f"  N endmembers           : {self.n_endmembers}")
            print(f"  N total pixels         : {self._n_pixels}")
            print(f"  N candidate pixels     : {self._n_candidates}")
            print(f"  Parallel workers       : {self._n_workers}")
            print(f"  Endmember threshold    : {self.endmember_threshold:.0%}")
            print(f"  Total combinations     : {total_combos:,}")
            print(f"  Sum-to-one constraint  : {self.constrain_sum}")
            print(f"  Non-negative constraint: {self.non_negative}")
            print(f"  Initial total RMS      : {self._current_total_rms:.6g}")
            print("-" * 60)

        # --- live progress figure ---
        _tracker = None
        update_progress_tracker = None  # ensure always defined in this scope
        if show_progress:
            try:
                from .plotting import create_progress_tracker, update_progress_tracker, _is_jupyter
            except ImportError as _e:
                import warnings as _w
                _w.warn(f"Progress tracking unavailable: {_e}", stacklevel=2)
            else:
                if _progress_tracker is not None:
                    # Pre-created on the main Qt thread — use directly.
                    _tracker = _progress_tracker
                else:
                    try:
                        _tracker = create_progress_tracker(self)
                        if _tracker.mode == "jupyter":
                            from IPython.display import display as _ipy_display
                            _ipy_display(_tracker.fig)
                        update_progress_tracker(_tracker, self)
                    except Exception as _e:
                        import warnings as _w
                        _w.warn(f"Could not create progress tracker: {_e}", stacklevel=2)
                        _tracker = None

        # Save initial state immediately — guarantees files exist before the loop
        # even if the run is interrupted before the first checkpoint_interval fires.
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
            if verbose:
                print(f"  [initial checkpoint → {checkpoint_path}]")
        if results_path is not None:
            self.save_results(results_path)
            if verbose:
                print(f"  [initial results → {results_path}]")

        _last_accepted = self._n_accepted  # detect new acceptances
        _accepted_since_save = 0
        _unaccepted_since_save = 0
        # Whether the last verbose print was an in-place heartbeat line (no
        # trailing newline yet) -- if so, anything printed next needs to
        # start on a fresh line first. See the printing block below.
        _mid_inplace_line = False

        try:
            while not self._is_converged and not self._stop_requested:
                if max_iterations is not None and self._n_iterations >= max_iterations:
                    self._is_converged = True
                    self._convergence_reason = f"max_iterations ({max_iterations})"
                    break

                self.step()

                newly_accepted = self._n_accepted > _last_accepted
                _last_accepted = self._n_accepted

                # Print on every new acceptance and every progress_interval
                # heartbeat. current_total_rms and best_total_rms are always
                # equal here: this search only ever accepts a step that
                # strictly improves on the current RMS, and the current RMS
                # never regresses, so every accepted step is -- by
                # construction -- also a new best. One RMS value is enough.
                if verbose and (newly_accepted or self._n_iterations % progress_interval == 0):
                    line = (
                        f"iter {self._n_iterations:7d} | "
                        f"accepted {self._n_accepted:5d} | "
                        f"no improvement {self._n_no_improvement:7d} | "
                        f"combos {len(self._visited):,} | "
                        f"RMS {self._current_total_rms:.6g}"
                    )
                    # \r returns to the start of the current terminal row and
                    # \033[K (ANSI "erase to end of line") clears whatever was
                    # there before, so a shorter new line doesn't leave stale
                    # trailing characters from a longer previous one. Padding
                    # to a fixed width (e.g. .ljust(100)) looks equivalent but
                    # isn't safe -- it forces a line wider than the actual
                    # content, and once that exceeds the terminal's column
                    # width it wraps onto a second row. \r then only returns
                    # to the start of *that* row, not the true start of the
                    # logical line, breaking the overwrite instead of fixing
                    # it. \033[K has no such width dependency.
                    if newly_accepted:
                        # A real, permanent line -- \r first to overwrite
                        # whatever in-place heartbeat is currently showing,
                        # then a trailing newline commits it to scrollback.
                        print("\r" + "* " + line + "\033[K")
                        _mid_inplace_line = False
                    else:
                        # In-place heartbeat: on a fast dataset these can
                        # fire many times a second with nothing new to
                        # report, scrolling the terminal too fast to read.
                        # Overwrite the same line instead (tqdm-style)
                        # rather than printing a new one every time.
                        print("\r" + "  " + line + "\033[K", end="", flush=True)
                        _mid_inplace_line = True

                # Update the live plot on accepted steps and at progress_interval
                # heartbeats.  Calling it every iteration floods the Qt JS queue
                # and causes "not responding" warnings.
                if _tracker is not None and update_progress_tracker is not None and (
                    newly_accepted or self._n_iterations % progress_interval == 0
                ):
                    update_progress_tracker(_tracker, self)

                # Periodic save: every 10 accepted or 100 unaccepted steps since last save
                if newly_accepted:
                    _accepted_since_save += 1
                else:
                    _unaccepted_since_save += 1

                if _accepted_since_save >= 10 or _unaccepted_since_save >= 100:
                    if verbose and _mid_inplace_line:
                        print()  # move off the in-place heartbeat line first
                        _mid_inplace_line = False
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
                        if verbose:
                            print(f"  [checkpoint → {checkpoint_path}]")
                    if results_path is not None:
                        self.save_results(results_path)
                        if verbose:
                            print(f"  [results → {results_path}]")
                    _accepted_since_save = 0
                    _unaccepted_since_save = 0

        except KeyboardInterrupt:
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
                print(f"\nInterrupted at iter {self._n_iterations} — "
                      f"checkpoint saved to: {checkpoint_path}")
            else:
                print(f"\nInterrupted at iter {self._n_iterations}.")
            if results_path is not None:
                self.save_results(results_path)
                print(f"  Partial results saved to: {results_path}")
            return self

        # Graceful stop: Ctrl+C was caught by the Qt main thread (--show_progress)
        # and forwarded via decomp._stop_requested rather than KeyboardInterrupt.
        if self._stop_requested:
            print(f"\nStopped at iter {self._n_iterations}.")
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
                if verbose:
                    print(f"  Checkpoint saved to: {checkpoint_path}")
            if results_path is not None:
                self.save_results(results_path)
                if verbose:
                    print(f"  Partial results saved to: {results_path}")
            return self

        # final update
        if _tracker is not None and update_progress_tracker is not None:
            update_progress_tracker(_tracker, self)

        if verbose:
            if _mid_inplace_line:
                print()  # move off the in-place heartbeat line first
            print("-" * 60)
            if self._convergence_reason == "exhausted_all_combinations":
                print(
                    "  Every possible endmember combination has been "
                    "evaluated -- this IS the true global optimum, "
                    f"found after {self._n_iterations} iterations."
                )
            else:
                print(f"  Converged after {self._n_iterations} iterations.")
            print(f"  Reason       : {self._convergence_reason}")
            print(f"  Accepted     : {self._n_accepted}")
            print(f"  Combinations : {len(self._visited)} evaluated")
            print(f"  Best total RMS: {self._best_total_rms:.6g}")
            print(f"  Best endmember indices: {self._best_em_indices.tolist()}")
            print("=" * 60)

        if results_path is not None:
            self.save_results(results_path)
            if verbose:
                print(f"  Results saved to: {results_path}")

        return self

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        """
        Save the current optimisation state to a pickle file.

        The file is written atomically (via a temp file + rename) so a crash
        mid-write will not corrupt the previous checkpoint.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"run_3em.ckpt"``).
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        state: Dict[str, Any] = {
            field: getattr(self, field) for field in _CHECKPOINT_FIELDS
        }
        state["_rng_state"] = self._rng.bit_generator.state
        state["_meta"] = {
            "n_endmembers": self.n_endmembers,
            "n_pixels": self._n_pixels,
            "n_candidates": self._n_candidates,
        }

        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def load_checkpoint(self, path) -> "EndmemberDecomposition":
        """
        Restore optimisation state from a checkpoint file created by
        :meth:`save_checkpoint`.

        The dataset and construction parameters (``n_endmembers``, etc.) must
        match those used when the checkpoint was saved; a ``ValueError`` is
        raised if they do not.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.

        Returns
        -------
        self
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        meta = state.pop("_meta", {})
        rng_state = state.pop("_rng_state", None)

        mismatches = []
        for key, expected in [
            ("n_endmembers", self.n_endmembers),
            ("n_pixels", self._n_pixels),
            ("n_candidates", self._n_candidates),
        ]:
            if key in meta and meta[key] != expected:
                mismatches.append(
                    f"  {key}: checkpoint={meta[key]}, current={expected}"
                )
        if mismatches:
            raise ValueError(
                "Checkpoint does not match this decomposition:\n"
                + "\n".join(mismatches)
            )

        for field, value in state.items():
            setattr(self, field, value)

        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

        return self

    def _build_results_dict(self) -> Dict[str, Any]:
        """
        Assemble the results dictionary in the format expected by
        :class:`~specdec.DecompositionResults`.
        """
        self._require_initialized()
        return {
            # Dataset — included so DecompositionResults needs no separate reload
            "dataset": self.dataset,

            # Core results
            "abundances":        self.abundances,
            "scale_factors":     self.scale_factors,
            "rms_errors":        self.rms_errors,
            "total_rms":         self.total_rms,
            "endmembers":        self.endmembers,
            "endmember_indices": self.endmember_indices,

            # Convergence tracking
            "n_iterations":          self.n_iterations,
            "n_accepted":            self.n_accepted,
            "is_converged":          self.is_converged,
            "convergence_reason":    self.convergence_reason,
            "accepted_rms_history":  self.accepted_rms_history,
            "em_index_history":      self.em_index_history,
            "tried_moves":           list(self._tried_moves),
            "n_combinations_tested": len(self.visited_combinations),

            # Resolved run parameters
            "params": {
                "n_endmembers":        self.n_endmembers,
                "n_jobs":              self.n_workers,
                "max_iterations":      getattr(self, "_max_iterations", None),
                "random_state":        self.random_state,
                "endmember_threshold": self.endmember_threshold,
                "free_sum":            self.free_sum,
                "smooth_endmembers":             self.smooth_endmembers,
                "endmember_smoothing_window":     self.endmember_smoothing_window,
                "endmember_smoothing_polyorder":  self.endmember_smoothing_polyorder,
            },
        }

    def save_results(self, path) -> None:
        """
        Save the current best results to a pickle file.

        The file is written atomically (via a temp file + rename) and can be
        loaded directly by :class:`~specdec.DecompositionResults`::

            from specdec import DecompositionResults
            r = DecompositionResults("my_results.pkl")

        The ``dataset`` object is included so no separate reload is needed.
        This method can be called at any point after :meth:`initialize` —
        including mid-run or after a ``KeyboardInterrupt`` — so partial results
        are never lost.  :meth:`run` accepts a *results_path* argument that
        calls this automatically on completion and on interrupt.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"ganymede_hst_3em_results.pkl"``).
        """
        import pickle
        from pathlib import Path

        results = self._build_results_dict()
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Results properties
    # ------------------------------------------------------------------

    def _require_initialized(self):
        if not self._is_initialized:
            raise RuntimeError(
                "No results yet — call initialize() or run() first."
            )

    @property
    def n_workers(self) -> int:
        """Resolved number of parallel workers (1 = sequential)."""
        return self._n_workers

    @property
    def endmembers(self) -> List[Pixel]:
        """Best endmember :class:`~specdec.Pixel` objects."""
        self._require_initialized()
        return [self._all_pixels[i] for i in self._best_em_indices]

    @property
    def endmember_indices(self) -> np.ndarray:
        """Global dataset indices of the best endmember pixels."""
        self._require_initialized()
        return self._best_em_indices.copy()

    @property
    def abundances(self) -> np.ndarray:
        """
        Abundance fractions for the best model.
        Shape ``(n_pixels, n_endmembers)``.
        Row order matches :attr:`Dataset.pixels`.
        """
        self._require_initialized()
        return self._best_abundances

    @property
    def scale_factors(self) -> Optional[np.ndarray]:
        """
        Per-pixel scale factors for the best model.  Shape ``(n_pixels,)``.

        Only populated when ``free_sum=True``; ``None`` otherwise.

        Each value is the sum of the raw NNLS weights for that pixel before
        normalisation.  The full spectral model is recovered as::

            modelled = scale_factors[:, None] * (abundances @ endmember_spectra)
        """
        self._require_initialized()
        return self._best_scale_factors

    @property
    def rms_errors(self) -> np.ndarray:
        """Per-pixel RMS residuals for the best model.  Shape ``(n_pixels,)``."""
        self._require_initialized()
        return self._best_rms_errors

    @property
    def total_rms(self) -> float:
        """Sum of all per-pixel RMS errors for the best model."""
        self._require_initialized()
        return self._best_total_rms

    @property
    def n_iterations(self) -> int:
        """Total perturbation steps (batches) executed."""
        return self._n_iterations

    @property
    def n_accepted(self) -> int:
        """Number of accepted perturbations (improvements)."""
        return self._n_accepted

    @property
    def is_converged(self) -> bool:
        """Whether the optimisation has converged."""
        return self._is_converged

    @property
    def convergence_reason(self) -> Optional[str]:
        """Human-readable reason for convergence, or ``None`` if not yet converged."""
        return self._convergence_reason

    @property
    def visited_combinations(self) -> Dict[frozenset, float]:
        """
        Mapping of each evaluated endmember combination (as a frozenset of
        global pixel indices) to its total summed RMS error.
        """
        return self._visited

    @property
    def accepted_rms_history(self) -> List[float]:
        """
        Total RMS at each *accepted* step (including initialisation).
        Useful for inspecting convergence behaviour.
        """
        return list(self._accepted_rms_history)

    @property
    def em_index_history(self) -> List[np.ndarray]:
        """
        Endmember global-index arrays at each accepted step (including
        initialisation).  Entry ``k`` is a copy of the endmember index array
        at the *k*-th accepted state, in the same order as
        :attr:`accepted_rms_history`.
        """
        return list(self._em_index_history)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def diagnose(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Re-run unmixing on the best endmembers and report solver diagnostics.

        Useful for understanding how many pixels used the fast vectorised path
        vs. the per-pixel NNLS fallback, and how many of those produced
        non-finite weights.

        Parameters
        ----------
        verbose : bool
            Print a formatted summary.  Default ``True``.

        Returns
        -------
        dict with keys:
            ``n_pixels``, ``n_batch_feasible``, ``n_nnls_fallback``,
            ``n_nonfinite_fallback``
        """
        self._require_initialized()
        diag: Dict[str, Any] = {}
        unmix_all(
            self._all_spectra,
            self._em_spectra(self._best_em_indices),
            constrain_sum=self.constrain_sum,
            non_negative=self.non_negative,
            minimization_fn=self.minimization_fn,
            _diagnostics=diag,
        )
        if verbose:
            n = diag.get("n_pixels", self._n_pixels)
            n_fast  = diag.get("n_batch_feasible", 0)
            n_nnls  = diag.get("n_nnls_fallback", 0)
            n_bad   = diag.get("n_nonfinite_fallback", 0)
            print("=" * 52)
            print("Unmixing diagnostics (best endmembers)")
            print("=" * 52)
            if not self.free_sum:
                print("  Mode               : FCLS (sum-to-one)")
            else:
                print("  Mode               : NNLS (free sum)")
            print(f"  Total pixels       : {n:,}")
            if "n_batch_feasible" in diag:
                print(f"  Fast batch path    : {n_fast:,}  ({n_fast/n:.1%})")
                print(f"  Per-pixel NNLS     : {n_nnls:,}  ({n_nnls/n:.1%})")
                print(f"  Non-finite fallback: {n_bad:,}  ({n_bad/n:.1%})")
            else:
                print("  (diagnostics only available for NNLS mode)")
            print("=" * 52)
        return diag

    def get_results(self) -> Dict[str, Any]:
        """
        Return a dictionary summarising the optimisation results.

        Keys
        ----
        endmember_indices, endmembers, abundances, rms_errors, total_rms,
        n_iterations, n_accepted, n_combinations_evaluated, is_converged,
        convergence_reason
        """
        self._require_initialized()
        return {
            "endmember_indices": self._best_em_indices.copy(),
            "endmembers": self.endmembers,
            "abundances": self._best_abundances.copy(),
            "rms_errors": self._best_rms_errors.copy(),
            "total_rms": self._best_total_rms,
            "n_iterations": self._n_iterations,
            "n_accepted": self._n_accepted,
            "n_combinations_evaluated": len(self._visited),
            "is_converged": self._is_converged,
            "convergence_reason": self._convergence_reason,
        }

    def __repr__(self) -> str:
        status = (
            "converged"
            if self._is_converged
            else ("running" if self._is_initialized else "not initialized")
        )
        rms_str = (
            f", best_total_rms={self._best_total_rms:.6g}"
            if self._best_total_rms is not None
            else ""
        )
        workers_str = f", n_workers={self._n_workers}" if self._n_workers > 1 else ""
        return (
            f"EndmemberDecomposition("
            f"n_endmembers={self.n_endmembers}, "
            f"n_pixels={self._n_pixels}, "
            f"status={status!r}"
            f"{rms_str}"
            f"{workers_str})"
        )
