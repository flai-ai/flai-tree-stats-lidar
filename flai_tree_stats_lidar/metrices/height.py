import numpy as np


def metrics_basic(z):
    """Basic height distribution statistics."""
    if len(z) == 0:
        return {k: np.nan for k in ['n', 'zmax', 'zmin', 'zmean', 'zvar', 'zsd', 'zcv', 'zskew', 'zkurt']}

    n = len(z)
    zmean = np.mean(z)
    zsd = np.std(z, ddof=1) if n > 1 else 0.0
    zvar = zsd ** 2

    result = {
        'n': n,
        'zmax': np.max(z),
        'zmin': np.min(z),
        'zmean': zmean,
        'zvar': zvar,
        'zsd': zsd,
        'zcv': (zsd / zmean) if zmean != 0 else np.nan,
    }

    if n > 2:
        m3 = np.mean((z - zmean) ** 3)
        result['zskew'] = m3 / (zsd ** 3) if zsd > 0 else np.nan
    else:
        result['zskew'] = np.nan

    if n > 3:
        m4 = np.mean((z - zmean) ** 4)
        result['zkurt'] = (m4 / (zsd ** 4)) - 3.0 if zsd > 0 else np.nan
    else:
        result['zkurt'] = np.nan

    return result


def metrics_percentiles(z):
    """Height percentiles: zq1, zq5, zq10, zq15, ..., zq95, zq99."""
    percentile_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    if len(z) == 0:
        return {f'zq{p}': np.nan for p in percentile_values}

    pcts = np.percentile(z, percentile_values)
    return {f'zq{p}': float(v) for p, v in zip(percentile_values, pcts)}


def metrics_percabove(z, thresholds=None):
    """Percentage of points above mean and given height thresholds."""
    if thresholds is None:
        thresholds = [2, 5]

    if len(z) == 0:
        result = {'pzabovemean': np.nan}
        for t in thresholds:
            result[f'pzabove{t}'] = np.nan
        return result

    n = len(z)
    result = {'pzabovemean': float(np.sum(z > np.mean(z)) / n * 100)}
    for t in thresholds:
        result[f'pzabove{t}'] = float(np.sum(z > t) / n * 100)

    return result


def metrics_dispersion(z, dz=1.0):
    """Dispersion metrics: IQR, MAD, CRR, entropy."""
    keys = ['ziqr', 'zMADmean', 'zMADmedian', 'CRR', 'zentropy']
    if len(z) == 0:
        return {k: np.nan for k in keys}

    q25, q75 = np.percentile(z, [25, 75])
    zmean = np.mean(z)
    zmedian = np.median(z)
    zmin = np.min(z)
    zmax = np.max(z)

    result = {
        'ziqr': float(q75 - q25),
        'zMADmean': float(np.mean(np.abs(z - zmean))),
        'zMADmedian': float(np.median(np.abs(z - zmedian))),
        'CRR': float((zmax - zmean) / (zmax - zmin)) if (zmax - zmin) > 0 else np.nan,
    }

    # Shannon entropy on height bins
    zrange = zmax - zmin
    if zrange > 0:
        bins = np.arange(zmin, zmax + dz, dz)
        counts, _ = np.histogram(z, bins=bins)
        proportions = counts / counts.sum()
        proportions = proportions[proportions > 0]
        result['zentropy'] = float(-np.sum(proportions * np.log(proportions)))
    else:
        result['zentropy'] = 0.0

    return result


def metrics_lmoments(z):
    """L-moments (L1-L4) and ratios (Lskew, Lkurt, Lcoefvar) via probability-weighted moments."""
    keys = ['L1', 'L2', 'L3', 'L4', 'Lskew', 'Lkurt', 'Lcoefvar']
    if len(z) == 0:
        return {k: np.nan for k in keys}

    n = len(z)
    if n < 4:
        return {k: np.nan for k in keys}

    z_sorted = np.sort(z)

    # Probability-weighted moments (PWMs) using unbiased estimators
    # b_r = (1/n) * sum_{i=r+1}^{n} C(i-1, r) / C(n-1, r) * x_{i:n}
    i_idx = np.arange(n, dtype=np.float64)

    b0 = np.mean(z_sorted)
    b1 = np.sum(i_idx * z_sorted) / (n * (n - 1))
    b2 = np.sum(i_idx * (i_idx - 1) * z_sorted) / (n * (n - 1) * (n - 2))
    b3 = np.sum(i_idx * (i_idx - 1) * (i_idx - 2) * z_sorted) / (n * (n - 1) * (n - 2) * (n - 3))

    L1 = b0
    L2 = 2 * b1 - b0
    L3 = 6 * b2 - 6 * b1 + b0
    L4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    result = {
        'L1': float(L1),
        'L2': float(L2),
        'L3': float(L3),
        'L4': float(L4),
        'Lskew': float(L3 / L2) if L2 != 0 else np.nan,
        'Lkurt': float(L4 / L2) if L2 != 0 else np.nan,
        'Lcoefvar': float(L2 / L1) if L1 != 0 else np.nan,
    }

    return result
