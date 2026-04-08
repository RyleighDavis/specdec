"""
specdec
=======

Iterative endmember decomposition for spatially resolved planetary spectra.

Typical workflow — single observation::

    from specdec import Pixel, Observation, Dataset, EndmemberDecomposition

    obs = Observation(observation_info={'slit_id': 1, 'disk_center_lon': 0.0})
    for wl, spec, coords, ea in my_data:
        obs.add_pixel(Pixel(wl, spec, coordinates=coords,
                            metadata={'emission_angle': ea}))

    ds = Dataset([obs])
    ds.exclude_by_metadata('emission_angle', '>', 60)

    decomp = EndmemberDecomposition(ds, n_endmembers=4)
    decomp.run()

    print(decomp.endmembers)   # list of Pixel objects
    print(decomp.abundances)   # (n_pixels, 4) array
    print(decomp.rms_errors)   # (n_pixels,) per-pixel RMS
    print(decomp.total_rms)    # scalar

Typical workflow — multiple observations (e.g. HST/STIS slit positions)::

    obs1 = Observation(observation_info={'slit_id': 1, 'disk_center_lon': 0.0})
    obs2 = Observation(observation_info={'slit_id': 2, 'disk_center_lon': 5.0})

    # … populate each observation with pixels …

    ds = Dataset([obs1, obs2])
    ds.exclude_by_metadata('emission_angle', '>', 60)

    decomp = EndmemberDecomposition(ds, n_endmembers=4)
    decomp.run()
"""

from .pixel import Pixel
from .dataset import Observation, Dataset
from .decomposition import EndmemberDecomposition
from .results import DecompositionResults
from .algorithms import (
    spectral_angle,
    spectral_angles_to_references,
    compute_rms,
    unmix_pixel,
    unmix_all,
)
from .plotting import (
    plot_endmember_spectra,
    plot_interactive_explorer,
    plot_abundance_simplex,
    plot_search_history,
    create_progress_tracker,
    update_progress_tracker,
    ProgressTracker,
)
from . import config

__version__ = "0.1.0"
__all__ = [
    "Pixel",
    "Observation",
    "Dataset",
    "EndmemberDecomposition",
    "DecompositionResults",
    "spectral_angle",
    "spectral_angles_to_references",
    "compute_rms",
    "unmix_pixel",
    "unmix_all",
    "plot_endmember_spectra",
    "plot_interactive_explorer",
    "plot_abundance_simplex",
    "plot_search_history",
    "create_progress_tracker",
    "update_progress_tracker",
    "ProgressTracker",
    "config",
]
