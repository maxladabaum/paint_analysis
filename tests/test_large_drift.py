from types import ModuleType, SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import numba

from drift_analysis import memory_bounded_aim
from paint_analysis_gui import _write_cached_dataframe, apply_drift_correction


def test_aim_processes_one_shift_at_a_time_without_changing_module():
    module = ModuleType('test_aim')
    module.lib = SimpleNamespace(ProgressDialog=object)
    calls = []

    def count(a, ac, b, bc):
        assert b.ndim == 1
        calls.append(b.copy())
        _, ai, bi = np.intersect1d(a, b, return_indices=True)
        return np.minimum(ac[ai], bc[bi]).sum()

    module._count_intersections = count
    exec('def _run_intersections_multithread(*args):\n    raise AssertionError("unbounded dispatcher called")\n'
         'def aim(*args):\n    return _run_intersections_multithread(*args)\n', module.__dict__)
    original = module._run_intersections_multithread
    bounded = memory_bounded_aim(module, str)
    a = np.arange(20)
    b = np.arange(4, 14)
    for box, shifts in ((3, np.arange(-4, 5)), (1, np.arange(-2, 3))):
        result = bounded(a, np.ones(20, int), b, np.full(10, 2), shifts, box)
        expected = [len(np.intersect1d(a, b + shift)) for shift in shifts]
        np.testing.assert_array_equal(result.ravel(), expected)
        assert result.shape == ((3, 3) if box == 3 else (5,))
    assert module._run_intersections_multithread is original
    assert module.lib.ProgressDialog is object
    assert bounded.__globals__['lib'].ProgressDialog is str
    np.testing.assert_array_equal(b, np.arange(4, 14))
    assert len(calls) == 14


def test_aim_supports_numba_kernel_with_explicit_shifts():
    module = ModuleType('test_new_aim')

    @numba.njit
    def count(a, ac, b, bc, shifts):
        values = np.zeros(len(shifts), dtype=np.int64)
        for s in range(len(shifts)):
            for j in range(len(b)):
                index = np.searchsorted(a, b[j] + shifts[s])
                if index < len(a) and a[index] == b[j] + shifts[s]:
                    values[s] += min(ac[index], bc[j])
        return values

    module._count_intersections = count
    exec('def _run_intersections(*args):\n    raise AssertionError("original dispatcher called")\n'
         'def aim(*args):\n    return _run_intersections(*args)\n', module.__dict__)
    original = module._run_intersections
    bounded = memory_bounded_aim(module)
    a = np.array([0, 2, 5, 8, 10])
    ac = np.array([3, 1, 4, 2, 5])
    b = np.array([1, 4, 7])
    bc = np.array([2, 6, 3])
    for box, shifts in ((3, np.arange(-4, 5)), (1, np.arange(-2, 3))):
        expected = []
        for shift in shifts:
            _, ai, bi = np.intersect1d(a, b + shift, return_indices=True)
            expected.append(np.minimum(ac[ai], bc[bi]).sum())
        result = bounded(a, ac, b, bc, shifts, box)
        np.testing.assert_array_equal(result.ravel(), expected)
        assert result.shape == ((3, 3) if box == 3 else (5,))
    assert module._run_intersections is original
    np.testing.assert_array_equal(b, [1, 4, 7])


def test_cache_writes_bounded_record_chunks_and_preserves_numeric_dtypes(tmp_path):
    frame = pd.DataFrame({'frame': np.arange(31, dtype=np.uint32),
                          'x': np.arange(31, dtype=np.float32),
                          'photons': np.arange(31, dtype=np.float64)})
    original = pd.DataFrame.to_records
    sizes = []

    def checked_records(self, *args, **kwargs):
        sizes.append(len(self))
        assert len(self) <= 4
        return original(self, *args, **kwargs)

    with h5py.File(tmp_path / 'cache.h5', 'w') as handle:
        with mock.patch.object(pd.DataFrame, 'to_records', checked_records):
            _write_cached_dataframe(handle, 'locs', frame, chunk_bytes=64)
        np.testing.assert_array_equal(handle['locs'][:], original(frame, index=False))
    assert max(sizes) == 4
    assert sum(sizes) == len(frame)


def test_drift_estimation_omits_extra_columns_but_returns_independent_complete_data():
    locs = pd.DataFrame({'frame': np.arange(8, dtype=np.int64),
                         'x': np.arange(8, dtype=np.float32),
                         'y': np.arange(8, dtype=np.float32),
                         'photons': np.full(8, 500, dtype=np.int32)}, index=np.arange(8)+20)
    before = locs.copy()

    def undrift(data, *args, **kwargs):
        assert list(data.columns) == ['frame', 'x', 'y']
        assert data.frame.dtype == np.uint32
        result = data.copy()
        result['x'] -= 1
        result['y'] += 2
        return pd.DataFrame({'x': np.ones(8), 'y': np.full(8, -2)}), result

    package = ModuleType('picasso')
    package.aim = SimpleNamespace()
    package.postprocess = SimpleNamespace(undrift=undrift)
    with mock.patch.dict('sys.modules', {'picasso': package}):
        corrected, _, _ = apply_drift_correction(
            locs, [{'Frames': 8, 'Pixelsize': 130}], 'rcc', 2, 7, 50)
    assert list(corrected.columns) == list(locs.columns)
    pd.testing.assert_index_equal(corrected.index, locs.index)
    pd.testing.assert_series_equal(corrected.frame, locs.frame)
    np.testing.assert_allclose(corrected.x, locs.x - 1)
    corrected.loc[20, 'photons'] = 0
    corrected.loc[20, 'x'] = 0
    pd.testing.assert_frame_equal(locs, before)
