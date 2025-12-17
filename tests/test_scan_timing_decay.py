import numpy as np
import pytest
from petpal.utils.scan_timing import calculate_frame_reference_time

def test_ref_time_no_decay_returns_midpoint():
    durations = np.array([5.0, 10.0])
    starts = np.array([0.0, 5.0])
    half_life = 1.0e8  # effectively no decay
    res = calculate_frame_reference_time(durations, starts, half_life)
    expected = starts + durations / 2.0
    assert np.allclose(res, expected, rtol=1e-2, atol=1e-3)


def test_ref_time_fast_decay_concentrates_near_start():
    durations = np.array([60.0, 30.0, 15.0])
    starts = np.array([0.0, 5.0, 10.0])
    half_life = 1e-6  # very fast decay
    res = calculate_frame_reference_time(durations, starts, half_life)
    delays = res - starts
    # For very fast decay, the reference time should be very close to frame start
    assert np.all(delays < durations * 0.01)


def test_ref_time_numeric_integration_agrees():
    # compare against numeric integral definition of weighted average time
    durations = np.array([5.0, 10.0, 60.0])
    starts = np.array([0.0, 5.0, 15.0])
    half_life = 1e4
    res = calculate_frame_reference_time(durations, starts, half_life)

    expected = []
    for T, s in zip(durations, starts):
        lam = np.log(2) / half_life
        t = np.linspace(0.0, T, 20001)
        w = np.exp(-lam * t)
        num = np.trapezoid(t * w, t)
        den = np.trapezoid(w, t)
        delay = num / den
        expected.append(s + delay)
    expected = np.asarray(expected)

    assert np.allclose(res, expected, rtol=1e-3, atol=1e-9)


def test_ref_time_vectorized_shape_and_broadcast():
    durations = np.array([10.0])
    starts = np.array([2.0])
    half_life = 1.0e3
    res = calculate_frame_reference_time(durations, starts, half_life)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1,)