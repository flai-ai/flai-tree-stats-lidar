import numpy as np
from scipy.stats import gaussian_kde


def metrics_kde(z, bw=2.0, n_peaks=4):
    """KDE-based peak detection on height distribution.

    Returns peak count and up to n_peaks peak elevations, values, and
    differences from the highest peak.
    """
    keys = ['kde_peaks_count']
    for i in range(1, n_peaks + 1):
        keys.extend([f'kde_peak{i}_elev', f'kde_peak{i}_value', f'kde_peak{i}_diff'])

    if len(z) < 3:
        return {k: np.nan for k in keys}

    zmin, zmax = np.min(z), np.max(z)
    if zmax == zmin:
        return {k: np.nan for k in keys}

    # Evaluate KDE on a fine grid
    grid = np.linspace(zmin, zmax, 512)
    try:
        kde = gaussian_kde(z, bw_method=bw / np.std(z, ddof=1) if np.std(z, ddof=1) > 0 else None)
    except Exception:
        return {k: np.nan for k in keys}

    density = kde(grid)

    # Find local maxima
    peak_mask = np.zeros(len(density), dtype=bool)
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peak_mask[i] = True

    peak_indices = np.where(peak_mask)[0]

    if len(peak_indices) == 0:
        # Use global maximum as single peak
        peak_indices = np.array([np.argmax(density)])

    # Sort peaks by density value (descending)
    peak_indices = peak_indices[np.argsort(density[peak_indices])[::-1]]

    result = {'kde_peaks_count': int(len(peak_indices))}

    max_elev = grid[peak_indices[0]]
    for i in range(1, n_peaks + 1):
        if i <= len(peak_indices):
            idx = peak_indices[i - 1]
            elev = grid[idx]
            result[f'kde_peak{i}_elev'] = float(elev)
            result[f'kde_peak{i}_value'] = float(density[idx])
            result[f'kde_peak{i}_diff'] = float(abs(elev - max_elev))
        else:
            result[f'kde_peak{i}_elev'] = np.nan
            result[f'kde_peak{i}_value'] = np.nan
            result[f'kde_peak{i}_diff'] = np.nan

    return result
