import numpy as np
from geometry import (DISTANCE_NAMES, LANDMARKS, coordinates_to_distances, orient_face,place_on_template, reconstruct, rigid_align)

#A plausible random face: 12 landmarks with the nose tip in front of the eye/mouth plane.
def random_face(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(12, 3)) * 30
    points[LANDMARKS.index("prn")] += [0, -10, 40]
    points[LANDMARKS.index("exL")] += [-30, 40, 0]
    points[LANDMARKS.index("exR")] += [30, 40, 0]
    points[LANDMARKS.index("chL")] += [-20, -40, 0]
    points[LANDMARKS.index("chR")] += [20, -40, 0]
    return points

def test_distance_names_have_66_pairs():
    assert len(DISTANCE_NAMES) == 66 and DISTANCE_NAMES[0] == "se-exL" and DISTANCE_NAMES[-1] == "chL-chR"

def test_reconstruction_reproduces_the_distances():
    distances = coordinates_to_distances(random_face())
    assert np.abs(coordinates_to_distances(reconstruct(distances)) - distances).max() < 1e-6
    
def test_orientation_removes_rotation_translation_and_mirror():
    face = random_face(1)
    rotation, _= np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    moved = face @ rotation + [5, -3, 8]
    mirrored = face * np.array([1, 1, -1])
    reference = orient_face(face)
    assert np.allclose(orient_face(moved), reference, atol=1e-6)
    assert np.allclose(orient_face(mirrored), reference, atol=1e-6)
    
def test_rigid_align_recovers_a_rotated_copy():
    face = random_face(3)
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    assert np.allclose(rigid_align(face @ rotation + 7, face), face, atol=1e-6)
    

def test_place_on_template_ignores_how_the_face_was_moved():
    face = random_face(5)
    template = orient_face(face) + 1.0
    rotation, _ = np.linalg.qr(np.random.default_rng(6).normal(size=(3, 3)))
    moved = face @ rotation + 3
    a = place_on_template(coordinates_to_distances(face), template)
    b = place_on_template(coordinates_to_distances(moved), template)
    
    assert np.allclose(a, b, atol=1e-6)

