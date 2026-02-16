import numpy as np


def metrics_canopy_density(z, n_intervals=10):
    """Cumulative canopy density profile (zpcum1..zpcum9).

    Divides the height range into n_intervals equal bands and returns
    the cumulative proportion of points below each breakpoint (excluding
    the last, which is always 1.0).
    """
    keys = [f'zpcum{i}' for i in range(1, n_intervals)]
    if len(z) == 0:
        return {k: np.nan for k in keys}

    zmin, zmax = np.min(z), np.max(z)
    if zmax == zmin:
        return {k: 1.0 for k in keys}

    breaks = np.linspace(zmin, zmax, n_intervals + 1)
    result = {}
    n = len(z)
    for i in range(1, n_intervals):
        result[f'zpcum{i}'] = float(np.sum(z <= breaks[i]) / n * 100)

    return result


def metrics_interval(z, intervals=None):
    """Proportion of points in each height band.

    Default intervals match lidRmetrics: [0, 0.15, 2, 5, 10, 20, 30].
    Returns pInterval_X_Y for each consecutive pair and pInterval_above_last.
    """
    if intervals is None:
        intervals = [0, 0.15, 2, 5, 10, 20, 30]

    keys = []
    for i in range(len(intervals) - 1):
        keys.append(f'pInterval_{intervals[i]}_{intervals[i+1]}')
    keys.append(f'pInterval_above_{intervals[-1]}')

    if len(z) == 0:
        return {k: np.nan for k in keys}

    n = len(z)
    result = {}
    for i in range(len(intervals) - 1):
        count = np.sum((z >= intervals[i]) & (z < intervals[i + 1]))
        result[f'pInterval_{intervals[i]}_{intervals[i+1]}'] = float(count / n * 100)

    result[f'pInterval_above_{intervals[-1]}'] = float(np.sum(z >= intervals[-1]) / n * 100)

    return result


def metrics_lad(z, dz=1.0, k=0.5, z0=2.0):
    """Leaf Area Density profile statistics using Beer-Lambert law.

    Parameters
    ----------
    z : array-like
        Normalized heights.
    dz : float
        Height bin size.
    k : float
        Light extinction coefficient.
    z0 : float
        Minimum height threshold (points below are excluded).
    """
    keys = ['lad_min', 'lad_max', 'lad_mean', 'lad_cv', 'lad_sum']
    if len(z) == 0:
        return {k: np.nan for k in keys}

    z_filt = z[z >= z0]
    if len(z_filt) == 0:
        return {k: np.nan for k in keys}

    zmin_bin = z0
    zmax_bin = np.max(z_filt)
    if zmax_bin <= zmin_bin:
        return {k: np.nan for k in keys}

    bins = np.arange(zmin_bin, zmax_bin + dz, dz)
    if len(bins) < 2:
        return {k: np.nan for k in keys}

    counts, _ = np.histogram(z_filt, bins=bins)
    n_total = len(z_filt)

    # Cumulative count from top down (number of points above each bin top)
    cum_above = np.cumsum(counts[::-1])[::-1]

    # Fraction of gap (points above / total) for Beer-Lambert
    lad_values = []
    for i in range(len(counts)):
        n_in = counts[i]
        n_above = cum_above[i]
        if n_above > 0 and n_in > 0:
            # LAD = -ln(1 - n_in / n_above) / (k * dz)
            ratio = n_in / n_above
            ratio = min(ratio, 0.999)  # clamp to avoid log(0)
            lad = -np.log(1.0 - ratio) / (k * dz)
            lad_values.append(lad)
        else:
            lad_values.append(0.0)

    lad_arr = np.array(lad_values)
    lad_arr = lad_arr[lad_arr > 0]

    if len(lad_arr) == 0:
        return {k: np.nan for k in keys}

    lad_mean = np.mean(lad_arr)
    lad_sd = np.std(lad_arr, ddof=1) if len(lad_arr) > 1 else 0.0

    return {
        'lad_min': float(np.min(lad_arr)),
        'lad_max': float(np.max(lad_arr)),
        'lad_mean': float(lad_mean),
        'lad_cv': float(lad_sd / lad_mean) if lad_mean > 0 else np.nan,
        'lad_sum': float(np.sum(lad_arr)),
    }
