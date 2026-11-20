import numpy as np
import pytest

from signspeak.preprocess import (
    landmarks_to_array,
    motion_energy,
    normalize_landmarks,
    normalize_sequence,
    resample_sequence,
)


def test_wrist_is_origin_and_scale_is_one_along_middle_mcp():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = (10, 20, 1)
    pts[9] = (10, 23, 1)
    pts[4] = (12, 20, 1)
    out = normalize_landmarks(pts)
    assert out.shape == (21, 3)
    assert np.allclose(out[0], 0)
    assert np.allclose(out[9, 1], 1.0, atol=1e-5)
    assert abs(float(np.linalg.norm(out[9, :2])) - 1) < 1e-4


def test_mirror_flips_relative_x():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[9] = (0, 1, 0)
    pts[4] = (0.5, 0.2, 0)
    plain = normalize_landmarks(pts)
    flipped = normalize_landmarks(pts, mirror=True)
    assert np.allclose(flipped[:, 0], -plain[:, 0])
    assert np.allclose(flipped[:, 1], plain[:, 1])


def test_flat_42_and_63_and_reject_bad_shape():
    xy = np.arange(42, dtype=np.float32)
    arr = landmarks_to_array(xy)
    assert arr.shape == (21, 3)
    assert np.all(arr[:, 2] == 0)
    xyz = np.arange(63, dtype=np.float32)
    assert landmarks_to_array(xyz).shape == (21, 3)
    with pytest.raises(ValueError):
        landmarks_to_array(np.arange(10))


def test_sequence_length_and_first_wrist():
    frames = []
    for i in range(10):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = (i, 0, 0)
        pts[9] = (i, 2, 0)
        frames.append(pts)
    seq = normalize_sequence(frames, length=30)
    assert seq.shape == (30, 21, 3)
    assert np.allclose(seq[0, 0], 0, atol=1e-5)
    assert motion_energy(seq) > 0
    same = resample_sequence(seq, 30)
    assert same.shape == seq.shape


def test_zero_z_clears_depth():
    pts = np.ones((21, 3), dtype=np.float32)
    pts[9, 1] = 3
    out = normalize_landmarks(pts, zero_z=True)
    assert np.allclose(out[:, 2], 0)
