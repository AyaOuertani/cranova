import numpy as np
import pytest

from skull_placement import SKULL_FILE, read_obj, scale_to_measurements, skull_landmarks

pytestmark = pytest.mark.skipif(not SKULL_FILE.exists(), reason="reference skull file not installed")


def test_scaled_skull_has_the_entered_measurements():
    vertices, _ = read_obj(SKULL_FILE)
    marks = skull_landmarks(vertices)
    scaled = scale_to_measurements(vertices, marks, gol=185.0, xcb=140.0, ufht=70.0)
    assert np.isclose(scaled[marks["glabella"], 2] - scaled[marks["opisthocranion"], 2], 185.0)
    assert np.isclose(scaled[marks["nasion"], 1] - scaled[marks["prosthion"], 1], 70.0)
    assert np.isclose(np.ptp(scaled[vertices[:, 1] > 0.2, 0]), 140.0)    # breadth of the vault