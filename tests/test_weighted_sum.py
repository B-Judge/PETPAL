import numpy as np
import pytest
from petpal.utils.math_lib import weighted_sum_computation
from petpal.utils.scan_timing import calculate_frame_reference_time
from petpal.preproc.decay_correction import calculate_frame_decay_factor

def test_weighted_sum_computation_all_ones_simple():
    # simple case: all voxels = 1, decay_correction = 1, frame_start[0]=0
    pet_series = np.ones((2, 2, 2, 3), dtype=float)
    frame_duration = np.array([10.0, 20.0, 30.0])
    half_life = 100.0
    frame_start = np.array([0.0, 10.0, 30.0])
    decay_correction = np.ones(3)

    decay_constant = np.log(2.0) / half_life
    image_total_duration = frame_duration.sum()
    total_decay = decay_constant * image_total_duration
    total_decay /= 1.0 - np.exp(-decay_constant * image_total_duration)
    total_decay /= np.exp(-decay_constant * frame_start[0])

    expected = np.full((2, 2, 2), total_decay)
    out = weighted_sum_computation(pet_series, frame_duration, half_life, frame_start, decay_correction)
    assert np.allclose(out, expected)

def test_weighted_sum_computation_manual_computation():
    # randomized small example, validate against manual numpy computation
    rng = np.random.default_rng(0)
    pet_series = rng.random((3, 2, 1, 4)).astype(float)
    frame_duration = np.array([5.0, 5.0, 10.0, 300.0])
    half_life = 50.0
    frame_start = np.array([1.0, 6.0, 11.0, 21.0])
    frame_ref_time = calculate_frame_reference_time(frame_duration=frame_duration,
                                                    frame_start=frame_start,
                                                    half_life=half_life)
    decay_correction = calculate_frame_decay_factor(frame_reference_time=frame_ref_time,
                                                    half_life=half_life)

    # manual expected computation following function logic
    decay_constant = np.log(2.0) / half_life
    image_total_duration = frame_duration.sum()
    total_decay = decay_constant * image_total_duration
    total_decay /= 1.0 - np.exp(-decay_constant * image_total_duration)
    total_decay /= np.exp(-decay_constant * frame_start[0])

    # compute weighted sum: sum_t(pet[...,t] * frame_duration[t] / decay_correction[t])
    scaled = pet_series * frame_duration / decay_correction  # broadcasting over last axis
    pet_series_sum_scaled = scaled.sum(axis=3)
    expected = pet_series_sum_scaled * total_decay / image_total_duration

    out = weighted_sum_computation(pet_series, frame_duration, half_life, frame_start, decay_correction)
    assert np.allclose(out, expected)