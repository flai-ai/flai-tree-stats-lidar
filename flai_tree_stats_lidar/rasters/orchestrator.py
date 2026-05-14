"""Orchestrate per-tile raster computation across many LAZ files.

Outer parallelism: ProcessPoolExecutor over tiles (so each tile runs in its
own Python process, releasing the GIL and isolating memory).
Inner parallelism: ThreadPoolExecutor over cells inside ``process_tile``.
"""
import csv
import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from .tile import process_tile

logger = logging.getLogger('flai')

LAZ_EXTENSIONS = ('.laz', '.las', '.LAZ', '.LAS')

FAILED_TXT = 'failed_tiles.txt'
FAILED_CSV = 'failed_tiles.csv'


def discover_pairs(lidar_dir: str, dem_dir):
    """Find LAZ files in ``lidar_dir`` and (optionally) pair them with
    same-basename DEM TIFs in ``dem_dir``.

    If ``dem_dir`` is ``None`` the DEM is computed on-the-fly per tile from
    its ground returns -- all LAZ files are emitted with ``dem_path=None``.

    Returns ``(pairs, missing)``: ``pairs`` is a list of ``(laz_path,
    dem_path or None)``; ``missing`` lists LAZ basenames that have no DEM
    when one was expected.
    """
    laz_files = []
    for ext in LAZ_EXTENSIONS:
        laz_files.extend(glob.glob(os.path.join(lidar_dir, f'*{ext}')))
    laz_files = sorted(set(laz_files))

    if dem_dir is None:
        return [(laz, None) for laz in laz_files], []

    pairs = []
    missing = []
    for laz in laz_files:
        base = os.path.splitext(os.path.basename(laz))[0]
        candidates = [
            os.path.join(dem_dir, f'{base}.tif'),
            os.path.join(dem_dir, f'{base}.TIF'),
            os.path.join(dem_dir, f'{base}.tiff'),
            os.path.join(dem_dir, f'{base}.TIFF'),
        ]
        dem = next((c for c in candidates if os.path.isfile(c)), None)
        if dem is None:
            missing.append(base)
        else:
            pairs.append((laz, dem))
    return pairs, missing


def _cap_blas_threads(n):
    """Cap BLAS / OpenMP / MKL thread pools to ``n`` for this process.

    Without this each subprocess can spawn cpu_count BLAS threads, so a
    workers=4 run on a 48-core box can balloon to ~200 threads competing
    on 48 cores. Setting env vars first helps libraries that read them at
    load time; threadpoolctl then reconfigures whatever's already loaded
    (OpenBLAS, MKL, OMP, BLIS) at runtime.
    """
    n = max(1, int(n))
    for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ.setdefault(var, str(n))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except ImportError:
        # env vars above are still in effect for libs that read them at load
        pass


def _worker(args):
    """Top-level (picklable) entry point for the process pool."""
    blas_threads = args.pop('_blas_threads', 1)
    _cap_blas_threads(blas_threads)
    # configure logging inside the worker process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s - %(message)s',
    )
    try:
        return process_tile(**args)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger('flai').exception(
            'tile failed: %s', args.get('laz_path'))
        return {
            'tile': os.path.splitext(os.path.basename(args['laz_path']))[0],
            'status': 'error',
            'error': repr(exc),
        }


def _make_progress(total, enable=True):
    """Try to return a tqdm progress bar; fall back to a no-op shim."""
    if not enable:
        return _NoopBar()
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, unit='tile', dynamic_ncols=True,
                    smoothing=0.1)
    except Exception:
        logger.info('tqdm not available -- falling back to plain logging')
        return _NoopBar()


class _NoopBar:
    def update(self, _n=1): pass
    def set_postfix(self, **_kwargs): pass
    def write(self, msg): logger.info(msg)
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *_a): self.close()


def _persist_failures(save_dir, failures, laz_lookup):
    """Write `failed_tiles.txt` (one LAZ path per line) and
    `failed_tiles.csv` (basename, laz_path, error) for downstream retry."""
    if not failures:
        # Clean up any stale reports from previous runs so they don't mislead
        for name in (FAILED_TXT, FAILED_CSV):
            p = os.path.join(save_dir, name)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return None, None

    txt_path = os.path.join(save_dir, FAILED_TXT)
    csv_path = os.path.join(save_dir, FAILED_CSV)
    with open(txt_path, 'w') as f:
        for tile_basename, _err in failures:
            laz = laz_lookup.get(tile_basename, '')
            f.write(f'{laz}\n')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tile', 'laz_path', 'error'])
        for tile_basename, err in failures:
            w.writerow([tile_basename, laz_lookup.get(tile_basename, ''), err])
    return txt_path, csv_path


def run(
    lidar_dir: str,
    dem_dir=None,
    save_dir: str = None,
    extract_groups=('dem', 'height', 'canopy', 'kde', 'voxel'),
    pixel_size: float = 10.0,
    workers: int = 4,
    cell_workers: int = 4,
    ground_class: int = 2,
    vegetation_classes=(3, 4, 5),
    min_points: int = 4,
    skip_existing: bool = True,
    dem_pixel_size: float = 0.5,
    idw_power: float = 2.0,
    idw_k: int = 12,
    progress: bool = True,
    crs_override=None,
    blas_threads: int = 1,
):
    if not os.path.isdir(lidar_dir):
        raise NotADirectoryError(f'lidar_dir does not exist: {lidar_dir}')
    if dem_dir is not None and not os.path.isdir(dem_dir):
        raise NotADirectoryError(f'dem_dir does not exist: {dem_dir}')
    if save_dir is None:
        raise ValueError('save_dir is required')
    os.makedirs(save_dir, exist_ok=True)

    start = datetime.now()
    pairs, missing = discover_pairs(lidar_dir, dem_dir)
    if missing:
        logger.warning('%d LAZ files without matching DEM: %s%s',
                       len(missing), missing[:5],
                       '...' if len(missing) > 5 else '')
    logger.info('Processing %d tile pairs with workers=%d, cell_workers=%d',
                len(pairs), workers, cell_workers)
    if not pairs:
        _persist_failures(save_dir, [], {})
        return {'pairs': 0, 'ok': 0, 'skipped': 0, 'failed': 0,
                'missing_dem': missing, 'failures': [],
                'failed_txt': None, 'failed_csv': None,
                'elapsed_s': 0.0}

    jobs = [
        dict(
            laz_path=laz, dem_path=dem, save_dir=save_dir,
            pixel_size=pixel_size, extract_groups=tuple(extract_groups),
            ground_class=ground_class,
            vegetation_classes=tuple(vegetation_classes),
            min_points=min_points, cell_workers=cell_workers,
            skip_existing=skip_existing,
            dem_pixel_size=dem_pixel_size,
            idw_power=idw_power, idw_k=idw_k,
            crs_override=crs_override,
            _blas_threads=blas_threads,
        )
        for laz, dem in pairs
    ]
    laz_lookup = {
        os.path.splitext(os.path.basename(laz))[0]: laz
        for laz, _ in pairs
    }

    ok = 0
    skipped = 0
    failed = 0
    failures = []  # list of (basename, error_string)

    bar = _make_progress(len(jobs), enable=progress)

    def _record(res):
        nonlocal ok, skipped, failed
        status = res.get('status')
        if status == 'ok':
            ok += 1
        elif status == 'skipped':
            skipped += 1
        else:
            failed += 1
            failures.append((res.get('tile', '?'),
                             res.get('error', f'status={status}')))
        bar.set_postfix(ok=ok, skip=skipped, fail=failed)
        bar.update(1)

    try:
        if workers <= 1:
            for job in jobs:
                res = _worker(job)
                _record(res)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_worker, job) for job in jobs]
                for fut in as_completed(futs):
                    _record(fut.result())
    finally:
        bar.close()

    elapsed = (datetime.now() - start).total_seconds()
    logger.info('Done. ok=%d skipped=%d failed=%d in %.1fs',
                ok, skipped, failed, elapsed)

    failed_txt, failed_csv = _persist_failures(save_dir, failures, laz_lookup)
    if failures:
        logger.warning('Wrote failure list to %s (%d entries)',
                       failed_txt, len(failures))

    return {
        'pairs': len(pairs),
        'ok': ok,
        'skipped': skipped,
        'failed': failed,
        'missing_dem': missing,
        'failures': failures,
        'failed_txt': failed_txt,
        'failed_csv': failed_csv,
        'elapsed_s': elapsed,
    }
