"""Registry of parameter keys produced by each metric group.

Calling the underlying metric functions with empty arrays returns a dict
populated with all keys (all NaN). We use that as the authoritative source
so the lists stay in sync with the implementations.
"""
import numpy as np

from ..metrices.dem import dem_params_at_location  # noqa: F401  (kept for ref)
from ..metrices.height import (
    metrics_basic,
    metrics_percentiles,
    metrics_percabove,
    metrics_dispersion,
    metrics_lmoments,
)
from ..metrices.canopy import (
    metrics_canopy_density,
    metrics_interval,
    metrics_lad,
)
from ..metrices.kde import metrics_kde
from ..metrices.voxel import metrics_rumple, metrics_voxels


SUPPORTED_GROUPS = ('dem', 'height', 'canopy', 'kde', 'voxel')

DEM_KEYS = ('elevation', 'slope', 'aspect', 'eastness', 'northness')


def _probe_keys(*calls):
    keys = []
    for fn, args in calls:
        keys.extend(fn(*args).keys())
    return tuple(keys)


def group_keys(group: str) -> tuple:
    """Return the ordered tuple of output parameter names for a group."""
    empty_z = np.empty(0, dtype=np.float64)
    empty_xy = (np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64))

    if group == 'dem':
        return DEM_KEYS

    if group == 'height':
        return _probe_keys(
            (metrics_basic, (empty_z,)),
            (metrics_percentiles, (empty_z,)),
            (metrics_percabove, (empty_z,)),
            (metrics_dispersion, (empty_z,)),
            (metrics_lmoments, (empty_z,)),
        )

    if group == 'canopy':
        return _probe_keys(
            (metrics_canopy_density, (empty_z,)),
            (metrics_interval, (empty_z,)),
            (metrics_lad, (empty_z,)),
        )

    if group == 'kde':
        return _probe_keys((metrics_kde, (empty_z,)))

    if group == 'voxel':
        return _probe_keys(
            (metrics_rumple, empty_xy),
            (metrics_voxels, empty_xy),
        )

    raise ValueError(f"Unknown metric group: {group!r}. "
                     f"Supported: {SUPPORTED_GROUPS}")


def collect_keys(groups: list) -> dict:
    """Return {group: (keys...)} for the requested groups, in order."""
    return {g: group_keys(g) for g in groups}


def flatten_keys(group_to_keys: dict) -> list:
    out = []
    for keys in group_to_keys.values():
        out.extend(keys)
    return out
