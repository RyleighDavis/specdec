# specdec

Iterative endmember decomposition for spatially resolved planetary spectra.

`specdec` implements a stochastic search that finds *N* representative "endmember" spectra within a dataset and linearly unmixes every pixel in terms of those endmembers. It was designed for HST/STIS long-slit spectra of icy satellites but works with any spatially resolved spectral dataset.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/RyleighDavis/specdec.git
```

To include optional plotting dependencies:

```bash
pip install "git+https://github.com/RyleighDavis/specdec.git#egg=specdec[plot]"
```

All optional extras:

| Extra | Packages added |
|-------|---------------|
| `plot` | matplotlib, cartopy, plotly |
| `fits` | astropy |
| `geo` | geopandas |
| `all` | all of the above |

## Quick start

```python
import numpy as np
from specdec import Pixel, Observation, Dataset, EndmemberDecomposition

# --- Build pixels ---
obs = Observation(observation_info={"slit_id": 1, "disk_center_lon": 0.0})

for wl, spec, corners, ea in my_data:
    px = Pixel(
        wavelengths=wl,
        spectrum=spec,
        coordinates=corners,          # list of (lon°W, lat°N) corner tuples
        metadata={"emission_angle": ea, "lon": centroid_lon, "lat": centroid_lat},
    )
    obs.add_pixel(px)

# --- Assemble dataset and optionally filter ---
ds = Dataset([obs])
ds.exclude_by_metadata("emission_angle", ">", 60)

# --- Run decomposition ---
decomp = EndmemberDecomposition(ds, n_endmembers=4)
decomp.run()

# --- Inspect results ---
results = decomp.results           # DecompositionResults object
print(results.endmembers)          # list of 4 Pixel objects
print(results.abundances.shape)    # (n_pixels, 4)
print(results.rms_errors)          # per-pixel RMS array
print(results.total_rms)           # scalar
```

## Core concepts

### Pixel

A `Pixel` holds:
- `wavelengths` / `spectrum` — 1-D arrays of equal length
- `coordinates` — list of ≥ 3 `(lon, lat)` corner tuples (positive-degrees-West, degrees-North), a shapely Polygon, or `None`
- `metadata` — arbitrary dict (e.g. emission angle, lat/lon centroid)
- `is_candidate` — whether this pixel may be chosen as an endmember

Non-finite spectral values are interpolated from neighbours on construction; a warning is issued. Pixels whose footprint crosses the antimeridian or 0°W/360°W boundary are automatically split into a `MultiPolygon`.

### Observation

An `Observation` groups pixels that share a common viewing geometry (one slit position, one IFU pointing, etc.). Multiple observations can be combined into a `Dataset`.

### Dataset

A `Dataset` holds one or more `Observation` objects and provides filtering methods:

```python
ds.exclude_by_metadata("emission_angle", ">", 60)
ds.exclude_by_metadata("lat", "<", -80)
ds.exclude_by_metadata("lon", "between", (150, 210))
```

Pixels can also be marked individually:

```python
my_pixel.exclude()    # mark ineligible
my_pixel.include()    # mark eligible again
```

### EndmemberDecomposition

```python
decomp = EndmemberDecomposition(
    dataset,
    n_endmembers=4,
    max_iterations=5000,
    n_no_improve=500,         # stop after this many consecutive non-improving steps
    endmember_threshold=0.50, # minimum abundance fraction to be a candidate replacement (default)
    n_jobs=4,                 # parallel candidate evaluation (1 = serial)
    checkpoint_path="run.pkl",
    checkpoint_interval=100,
    verbose=True,
)
decomp.run()
```

The algorithm:
1. Initialise *N* endmembers via K-means + spectral angle mapping (SAM).
2. Unmix every pixel as a non-negative linear combination of the current endmembers.
3. Perturb: for each endmember in turn, randomly pick a replacement from pixels that carry ≥ `endmember_threshold` (default 50%) abundance of that component.
4. Accept the replacement if the new total RMS is strictly lower.
5. Repeat until convergence (exhausted combinations, no-improvement plateau, or `max_iterations`).

Resuming from a checkpoint:

```python
decomp = EndmemberDecomposition(ds, n_endmembers=4, checkpoint_path="run.pkl")
decomp.resume()
```

### DecompositionResults

```python
results = decomp.results        # or load from file:
from specdec import DecompositionResults
results = DecompositionResults.load("run_results.pkl")
```

Key attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `endmembers` | `list[Pixel]` | The *N* best-fit endmember pixels |
| `abundances` | `ndarray (n_px, N)` | Linear mixing fractions (≥ 0, sum ≤ 1) |
| `scale_factors` | `ndarray (n_px,)` | Per-pixel overall scale applied before unmixing |
| `rms_errors` | `ndarray (n_px,)` | Per-pixel RMS residual |
| `total_rms` | `float` | Mean RMS across all pixels |
| `modelled_spectra` | `ndarray (n_px, n_wl)` | Reconstructed spectra |
| `pixels` | `list[Pixel]` | All pixels in the dataset |

## Plotting

Requires the `plot` extra (`pip install "specdec[plot]"`).

```python
# Endmember spectra
results.plot_endmember_spectra()

# Abundance maps (one panel per endmember)
results.plot_abundance_map()

# Interactive map — click a cell to view its spectrum
results.plot_interactive_explorer()

# Simplex projection of abundances
results.plot_abundance_simplex()

# Single pixel: observed vs modelled
results.plot_pixel(pixel_id=42)

# Residual spectrum for one pixel
results.plot_residual_spectrum(pixel_id=42)

# Optimisation search history
results.plot_search_history()
```

All plotting functions return matplotlib `Figure` objects (or display inline in Jupyter) and accept optional keyword arguments for customisation — see the function docstrings for details.

### Interactive explorer

`plot_interactive_explorer()` starts a lightweight local HTTP server and renders a Plotly heatmap coloured by dominant endmember. Clicking a 1°×1° cell fetches the pixel's observed and modelled spectra on demand, so the initial render is fast even for large datasets.

## Standalone script

The `run_decomposition.py` script (found alongside the tutorial notebook) provides a command-line interface for common workflows:

```bash
python run_decomposition.py \
    --data_file ganymede_data.pkl \
    --n_endmembers 4 \
    --max_iterations 5000 \
    --n_jobs 4 \
    --output_dir results/
```

Run `python run_decomposition.py --help` for the full option list.

## Dependencies

| Package | Min version | Notes |
|---------|-------------|-------|
| numpy | 1.21 | |
| scipy | 1.7 | |
| scikit-learn | 1.0 | K-means initialisation |
| joblib | 1.1 | Parallel evaluation |
| shapely | 2.0 | Pixel footprint geometry |
| matplotlib | 3.5 | `plot` extra |
| cartopy | 0.21 | `plot` extra |
| plotly | 5.0 | `plot` extra |
| astropy | 5.0 | `fits` extra |
| geopandas | 0.12 | `geo` extra |

## License

MIT — see [LICENSE](LICENSE).
