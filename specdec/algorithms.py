"""
Core spectral algorithms: SAM, RMS, linear unmixing, K-means initialization.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Callable

from scipy.optimize import nnls, minimize
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Spectral Angle Mapper (SAM)
# ---------------------------------------------------------------------------


def spectral_angle(reference: np.ndarray, spectrum: np.ndarray) -> float:
    """
    Compute the Spectral Angle Mapper (SAM) angle between two spectra.

    .. math::

        \\theta = \\arccos\\!\\left(
            \\frac{\\mathbf{t} \\cdot \\mathbf{r}}{\\|\\mathbf{t}\\|\\,\\|\\mathbf{r}\\|}
        \\right)

    Parameters
    ----------
    reference : array-like, shape (n_wavelengths,)
        Reference spectrum **t** (e.g. a K-means cluster centre).
    spectrum : array-like, shape (n_wavelengths,)
        Pixel spectrum **r**.

    Returns
    -------
    float
        SAM angle in radians ∈ [0, π/2].
    """
    t = np.asarray(reference, dtype=float)
    r = np.asarray(spectrum, dtype=float)

    norm_t = np.linalg.norm(t)
    norm_r = np.linalg.norm(r)

    if norm_t == 0.0 or norm_r == 0.0:
        return np.pi / 2.0  # undefined → treat as orthogonal

    cos_theta = np.dot(t, r) / (norm_t * norm_r)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def spectral_angles_to_references(
    spectra: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """
    Compute SAM angles between every pixel spectrum and every reference.

    Parameters
    ----------
    spectra : ndarray, shape (n_pixels, n_wavelengths)
    references : ndarray, shape (n_refs, n_wavelengths)

    Returns
    -------
    ndarray, shape (n_pixels, n_refs)
        SAM angles in radians.
    """
    s = np.asarray(spectra, dtype=float)
    r = np.asarray(references, dtype=float)

    norms_s = np.linalg.norm(s, axis=1, keepdims=True)
    norms_r = np.linalg.norm(r, axis=1, keepdims=True)

    # Avoid division by zero
    norms_s = np.where(norms_s == 0.0, 1.0, norms_s)
    norms_r = np.where(norms_r == 0.0, 1.0, norms_r)

    s_norm = s / norms_s           # (n_pixels, n_wl)
    r_norm = r / norms_r           # (n_refs,   n_wl)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dots = s_norm @ r_norm.T   # (n_pixels, n_refs)
    return np.arccos(np.clip(dots, -1.0, 1.0))


# ---------------------------------------------------------------------------
# RMS error
# ---------------------------------------------------------------------------


def compute_rms(observed: np.ndarray, modeled: np.ndarray) -> float:
    """
    Root-mean-square error between two spectra.

    Parameters
    ----------
    observed : array-like, shape (n_wavelengths,)
    modeled : array-like, shape (n_wavelengths,)

    Returns
    -------
    float
    """
    obs = np.asarray(observed, dtype=float)
    mod = np.asarray(modeled, dtype=float)
    return float(np.sqrt(np.mean((obs - mod) ** 2)))


# ---------------------------------------------------------------------------
# Linear unmixing
# ---------------------------------------------------------------------------


def unmix_pixel(
    spectrum: np.ndarray,
    endmember_spectra: np.ndarray,
    constrain_sum: bool = True,
    non_negative: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Decompose one pixel spectrum as a linear combination of endmember spectra.

    Parameters
    ----------
    spectrum : array-like, shape (n_wavelengths,)
        Observed pixel spectrum **r**.
    endmember_spectra : array-like, shape (n_endmembers, n_wavelengths)
        One endmember per row.
    constrain_sum : bool
        Enforce ``sum(abundances) == 1``.  Default ``True``.
    non_negative : bool
        Enforce ``abundances >= 0``.  Default ``True``.

    Returns
    -------
    abundances : ndarray, shape (n_endmembers,)
    rms : float
        RMS residual of the fit.

    Notes
    -----
    * ``constrain_sum=True, non_negative=True``  → SLSQP (fully constrained)
    * ``constrain_sum=False, non_negative=True`` → NNLS
    * ``constrain_sum=False, non_negative=False``→ unconstrained least squares
    """
    r = np.asarray(spectrum, dtype=float)
    E = np.asarray(endmember_spectra, dtype=float)  # (n_em, n_wl)
    n_em = E.shape[0]

    if constrain_sum:
        # Quadratic programming via SLSQP
        def objective(a: np.ndarray) -> float:
            residual = r - E.T @ a
            return 0.5 * float(np.dot(residual, residual))

        def jac(a: np.ndarray) -> np.ndarray:
            return E @ (E.T @ a - r)

        constraints = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
        bounds = [(0.0, None)] * n_em if non_negative else [(None, None)] * n_em
        a0 = np.full(n_em, 1.0 / n_em)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = minimize(
                objective,
                a0,
                jac=jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 2000},
            )
        abundances = result.x

    elif non_negative:
        # Non-negative least squares (no sum constraint)
        abundances, _ = nnls(E.T, r)

    else:
        # Unconstrained ordinary least squares
        abundances, _, _, _ = np.linalg.lstsq(E.T, r, rcond=None)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        modeled = E.T @ abundances
    rms = compute_rms(r, modeled)
    return abundances, rms


def _solve_eq_constrained_batch(
    B: np.ndarray,
    G: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Solve the sum-to-one equality-constrained QP for all pixels simultaneously.

    For each pixel i, finds ``a`` minimising ``||E^T a - r_i||^2`` subject to
    ``sum(a) == 1``, ignoring the non-negativity constraint.

    Uses the Lagrange-multiplier formula:

        a_i = G^{-1} b_i − μ_i G^{-1} 1
        μ_i = (1^T G^{-1} b_i − 1) / (1^T G^{-1} 1)

    where ``b_i = E r_i`` (row ``i`` of ``B``) and ``G = E E^T``.

    Parameters
    ----------
    B : ndarray, shape (n_pixels, n_em) — pre-computed ``S @ E.T``
    G : ndarray, shape (n_em, n_em)    — pre-computed ``E @ E.T``

    Returns
    -------
    A : ndarray, shape (n_pixels, n_em) or ``None`` on numerical failure
        May contain negative entries (caller applies non-negativity projection).
    G_inv : ndarray, shape (n_em, n_em)
    """
    n_em = G.shape[0]
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)

    ones = np.ones(n_em)
    G_inv_1 = G_inv @ ones            # (n_em,)
    denom = float(ones @ G_inv_1)
    if abs(denom) < 1e-14:
        return None, G_inv

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        D = (G_inv @ B.T).T               # (n_pixels, n_em)  G^{-1} b_i per pixel
        mu = (D @ ones - 1.0) / denom    # (n_pixels,)
        A = D - mu[:, None] * G_inv_1    # (n_pixels, n_em)
    return A, G_inv


def _fcls_pixel(b: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Active-set FCLS for a single pixel.

    Solves ``min ||E^T a - r||^2  s.t.  sum(a)==1, a>=0`` using an exact
    active-set method (at most ``n_em`` iterations, no scipy overhead).

    Parameters
    ----------
    b : ndarray, shape (n_em,) — pre-computed ``E @ r`` for this pixel
    G : ndarray, shape (n_em, n_em) — ``E @ E.T``

    Returns
    -------
    ndarray, shape (n_em,)
    """
    n_em = G.shape[0]
    free: List[int] = list(range(n_em))

    for _ in range(n_em):
        if len(free) == 1:
            a = np.zeros(n_em)
            a[free[0]] = 1.0
            return a

        f = np.array(free)
        G_f = G[np.ix_(f, f)]
        try:
            G_f_inv = np.linalg.inv(G_f)
        except np.linalg.LinAlgError:
            G_f_inv = np.linalg.pinv(G_f)

        ones_f = np.ones(len(f))
        G_f_inv_1 = G_f_inv @ ones_f
        denom_f = float(ones_f @ G_f_inv_1)
        if abs(denom_f) < 1e-14:
            break

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            d_f = G_f_inv @ b[f]
            mu_f = (float(ones_f @ d_f) - 1.0) / denom_f
            a_f = d_f - mu_f * G_f_inv_1

        if np.all(a_f >= -1e-9):
            a = np.zeros(n_em)
            a[f] = np.maximum(a_f, 0.0)
            return a

        # Fix the most negative free variable to 0
        worst = int(np.argmin(a_f))
        free.pop(worst)

    # Fallback: put all weight on the endmember most correlated with this pixel
    a = np.zeros(n_em)
    a[int(np.argmax(b))] = 1.0
    return a


def unmix_all(
    spectra: np.ndarray,
    endmember_spectra: np.ndarray,
    constrain_sum: bool = True,
    non_negative: bool = True,
    minimization_fn: Optional[Callable] = None,
    _diagnostics: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Decompose every pixel spectrum as a linear combination of endmember spectra.

    Parameters
    ----------
    spectra : ndarray, shape (n_pixels, n_wavelengths)
    endmember_spectra : ndarray, shape (n_endmembers, n_wavelengths)
    constrain_sum : bool
    non_negative : bool
    minimization_fn : callable, optional
        Custom per-pixel solver.  Signature::

            fn(spectrum, endmember_spectra) -> (abundances, rms)

        When supplied, *constrain_sum* and *non_negative* are ignored.
    _diagnostics : dict, optional
        If provided, the following keys are populated in-place:

        * ``n_pixels`` — total number of pixels
        * ``n_batch_feasible`` — pixels solved by the fast vectorised path
        * ``n_nnls_fallback`` — pixels that needed a per-pixel NNLS call
        * ``n_nonfinite_fallback`` — pixels where NNLS returned non-finite
          values and the clamped unconstrained solution was used instead

    Returns
    -------
    abundances : ndarray, shape (n_pixels, n_endmembers)
    rms_errors : ndarray, shape (n_pixels,)
        Per-pixel RMS residuals.
    total_rms : float
        Sum of all per-pixel RMS values.

    Notes
    -----
    All standard constraint combinations are solved with vectorised linear
    algebra (a single matrix multiply path handles the vast majority of
    pixels). Per-pixel fallback is only used for the custom *minimization_fn*
    or for the small fraction of FCLS pixels whose sum-=1 solution has
    negatives and needs active-set refinement.
    """
    S = np.asarray(spectra, dtype=float)
    E = np.asarray(endmember_spectra, dtype=float)
    n_pixels = S.shape[0]
    n_em = E.shape[0]

    abundances = np.zeros((n_pixels, n_em), dtype=float)

    # ── Custom solver ────────────────────────────────────────────────────
    if minimization_fn is not None:
        rms_errors = np.zeros(n_pixels, dtype=float)
        for i in range(n_pixels):
            abundances[i], rms_errors[i] = minimization_fn(S[i], E)
        return abundances, rms_errors, float(np.sum(rms_errors))

    # ── Precompute Gram matrix and pixel projections ──────────────────────
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        G = E @ E.T      # (n_em, n_em)
        B = S @ E.T      # (n_pixels, n_em)  — E r_i for every pixel i at once

    # ── Unconstrained batch LS ───────────────────────────────────────────
    if not constrain_sum and not non_negative:
        # min ||E^T a - r||^2  →  a = G^{-1} b  for each pixel
        # Batch: A = B G^{-T}  (G symmetric → G^{-T} = G^{-1})
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                abundances = np.linalg.solve(G, B.T).T
        except np.linalg.LinAlgError:
            abundances = (np.linalg.pinv(G) @ B.T).T

    # ── NNLS (no sum constraint, non-negative) ───────────────────────────
    elif not constrain_sum and non_negative:
        # Fast path: unconstrained solve gives the NNLS solution for any pixel
        # whose result is already non-negative (KKT conditions satisfied).
        # Only the minority with negatives needs the full per-pixel nnls call.
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                A_cand = np.linalg.solve(G, B.T).T   # (n_pixels, n_em)
        except np.linalg.LinAlgError:
            A_cand = (np.linalg.pinv(G) @ B.T).T

        feasible = np.all(A_cand >= -1e-9, axis=1)
        abundances[feasible] = np.maximum(A_cand[feasible], 0.0)
        infeasible_idx = np.where(~feasible)[0]
        n_nonfinite = 0
        import warnings as _warnings
        for i in infeasible_idx:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"), \
                 _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                result, _ = nnls(E.T, S[i])
            # If nnls produced non-finite weights (ill-conditioned pixel),
            # fall back to the clamped unconstrained solution.
            if np.all(np.isfinite(result)):
                abundances[i] = result
            else:
                abundances[i] = np.maximum(A_cand[i], 0.0)
                n_nonfinite += 1
        if _diagnostics is not None:
            _diagnostics["n_pixels"]           = n_pixels
            _diagnostics["n_batch_feasible"]   = int(feasible.sum())
            _diagnostics["n_nnls_fallback"]    = len(infeasible_idx)
            _diagnostics["n_nonfinite_fallback"] = n_nonfinite

    # ── Equality-constrained LS (sum=1, free sign) ───────────────────────
    elif constrain_sum and not non_negative:
        A_cand, _ = _solve_eq_constrained_batch(B, G)
        if A_cand is None:
            # Numerical fallback
            for i in range(n_pixels):
                abundances[i], _ = unmix_pixel(S[i], E,
                                               constrain_sum=True,
                                               non_negative=False)
        else:
            abundances = A_cand

    # ── FCLS (sum=1 + non-negative) — default, hot path ─────────────────
    else:
        # Step 1: solve sum=1 QP for ALL pixels at once (vectorised)
        A_cand, G_inv = _solve_eq_constrained_batch(B, G)

        if A_cand is None:
            # Gram matrix is singular — fall back to per-pixel
            for i in range(n_pixels):
                abundances[i], _ = unmix_pixel(S[i], E,
                                               constrain_sum=True,
                                               non_negative=True)
        else:
            # Step 2: pixels whose solution is already non-negative → done
            feasible = np.all(A_cand >= -1e-9, axis=1)
            abundances[feasible] = np.maximum(A_cand[feasible], 0.0)

            # Step 3: active-set refinement for the infeasible minority
            for i in np.where(~feasible)[0]:
                abundances[i] = _fcls_pixel(B[i], G)

    # ── Compute RMS vectorised ────────────────────────────────────────────
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        modeled = abundances @ E              # (n_pixels, n_wl)
    rms_errors = np.sqrt(np.mean((S - modeled) ** 2, axis=1))

    return abundances, rms_errors, float(np.sum(rms_errors))


# ---------------------------------------------------------------------------
# K-means + SAM initialisation
# ---------------------------------------------------------------------------


def initialize_endmembers_kmeans(
    spectra: np.ndarray,
    n_endmembers: int,
    pixel_indices: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick initial endmember pixel indices using K-means clustering + SAM.

    Algorithm
    ---------
    1. Run K-means on the candidate spectra to obtain *n_endmembers* cluster
       centres.
    2. For each cluster centre, select the candidate pixel with the smallest
       SAM angle as the initial endmember.

    Parameters
    ----------
    spectra : ndarray, shape (n_candidates, n_wavelengths)
        Spectra of candidate pixels only.
    n_endmembers : int
    pixel_indices : ndarray, shape (n_candidates,), optional
        Mapping from row index in *spectra* to a global pixel index.
        If ``None``, uses ``0 … n_candidates-1``.
    random_state : int, optional
        Random seed forwarded to :class:`~sklearn.cluster.KMeans`.

    Returns
    -------
    indices : ndarray, shape (n_endmembers,)
        Global pixel indices of the selected initial endmembers.
    cluster_centers : ndarray, shape (n_endmembers, n_wavelengths)
        K-means cluster centres, ordered to match *indices*.
    """
    n_candidates = spectra.shape[0]
    if pixel_indices is None:
        pixel_indices = np.arange(n_candidates)

    if n_endmembers > n_candidates:
        raise ValueError(
            f"n_endmembers ({n_endmembers}) exceeds the number of "
            f"candidate pixels ({n_candidates})."
        )

    # Drop any rows with NaN/Inf before fitting (safety net for bad input data)
    valid_mask = np.all(np.isfinite(spectra), axis=1)
    if not np.all(valid_mask):
        import warnings as _warnings
        _warnings.warn(
            f"{(~valid_mask).sum()} candidate spectra contain NaN/Inf and were "
            "excluded from K-means initialisation. Consider filtering them out "
            "before creating the Dataset.",
            stacklevel=3,
        )
        fit_spectra = spectra[valid_mask]
        fit_pixel_indices = pixel_indices[valid_mask]
    else:
        fit_spectra = spectra
        fit_pixel_indices = pixel_indices

    kmeans = KMeans(n_clusters=n_endmembers, random_state=random_state, n_init=10)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        kmeans.fit(fit_spectra)
    centers = kmeans.cluster_centers_  # (n_endmembers, n_wl)

    # SAM angles: (n_candidates, n_endmembers)
    angles = spectral_angles_to_references(fit_spectra, centers)

    selected_local: List[int] = []
    used: set = set()

    for j in range(n_endmembers):
        # Sort candidates by SAM angle to this centre
        order = np.argsort(angles[:, j])
        for idx in order:
            if idx not in used:
                selected_local.append(int(idx))
                used.add(idx)
                break
        else:
            # All candidates already assigned (shouldn't happen normally)
            # Fall back: pick the first unused
            for idx in range(n_candidates):
                if idx not in used:
                    selected_local.append(idx)
                    used.add(idx)
                    break

    # centers[j] is the K-means centre for cluster j, which maps to endmember j
    return fit_pixel_indices[np.array(selected_local)], centers


# Avoid name clash at module level
from typing import List  # noqa: E402  (used inside initialize_endmembers_kmeans)


# ---------------------------------------------------------------------------
# Parallel evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_combination(
    all_spectra: np.ndarray,
    em_indices: np.ndarray,
    constrain_sum: bool,
    non_negative: bool,
    minimization_fn,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate one endmember combination by unmixing all pixels against it.

    Defined at module level so it can be pickled by joblib's loky / multiprocessing
    backends.  Returns the same (abundances, rms_errors, total_rms) triple as
    :func:`unmix_all`.

    Notes
    -----
    When a custom *minimization_fn* is used with ``n_jobs > 1`` it must itself
    be picklable (i.e. a module-level function — not a lambda or closure).
    """
    em_spectra = all_spectra[em_indices]
    return unmix_all(
        all_spectra,
        em_spectra,
        constrain_sum=constrain_sum,
        non_negative=non_negative,
        minimization_fn=minimization_fn,
    )
