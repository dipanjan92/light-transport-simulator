import taichi as ti
from taichi.math import vec3, normalize

from pbrt.parse_utils import py_normalize


def extract_camera_from_scene(scene_dict):
    """Extracts eye, center, and up from PBRT transformation matrix in scene_dict."""
    # Extract the 4×4 transformation matrix
    transform_data = scene_dict["Transform"][0]["properties"]["matrix"]
    camera_data = scene_dict["Camera"][0]["properties"]
    fov = float(camera_data.get('fov', 45.0)[0])

    # PBRT matrices are in **column-major order**, so we extract rows properly
    M = ti.Matrix([
        [transform_data[0], transform_data[1], transform_data[2], transform_data[3]],
        [transform_data[4], transform_data[5], transform_data[6], transform_data[7]],
        [transform_data[8], transform_data[9], transform_data[10], transform_data[11]],
        [transform_data[12], transform_data[13], transform_data[14], transform_data[15]]
    ])

    # Extract camera parameters
    eye = vec3(M[0, 3], M[1, 3], M[2, 3])  # Camera position
    forward = -py_normalize(vec3(M[0, 2], M[1, 2], M[2, 2]))  # Camera forward direction (PBRT convention)
    center = eye + forward  # Look-at point
    up = py_normalize(vec3(M[0, 1], M[1, 1], M[2, 1]))  # Camera up vector

    return eye, center, up, fov