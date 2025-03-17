# parse_utils.py
"""
This module provides utility functions for parsing parameters and performing mathematical operations
for a physically-based rendering system. It includes functions for extracting numbers from strings,
cleaning and parsing parameter values, converting values to Taichi vectors, and generating transformation matrices.
"""

import re
import taichi as ti
import numpy as np
from taichi.math import vec3

def extract_numbers(s):
    """
    Extract all numeric substrings from the given string using a regular expression.

    Args:
        s (str): The input string from which to extract numbers.

    Returns:
        list: A list of numeric substrings found in the input string.
    """
    number_pattern = r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?'
    return re.findall(number_pattern, s)

def clean_brackets(s):
    """
    Clean the input string by stripping whitespace and removing surrounding brackets or quotes.

    Args:
        s (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    return s.strip().strip("[]").strip('"')

def parse_value(param_type, raw_value):
    """
    Parse a raw string value into a specific type based on the provided parameter type.

    Args:
        param_type (str): The type to parse the value into (e.g., 'float', 'integer', 'vector', etc.).
        raw_value (str): The raw string value to parse.

    Returns:
        The parsed value in the appropriate type, or None if parsing fails.
    """
    cleaned = clean_brackets(raw_value)
    if param_type in ["float", "integer", "point3", "point2", "vector", "vector3", "rgb", "spectrum", "point", "normal"]:
        tokens = extract_numbers(cleaned)
        if param_type == "normal":
            tokens = [tok for tok in tokens if tok != "."]
        if not tokens:
            return None
    else:
        tokens = cleaned.split()
    try:
        if param_type == "float":
            return float(tokens[0])
        elif param_type == "integer":
            if len(tokens) > 1:
                return [int(float(tok)) for tok in tokens]
            else:
                return int(float(tokens[0]))
        elif param_type in ["point3", "point2", "point"]:
            return tuple(float(tok) for tok in tokens)
        elif param_type in ["vector", "vector3", "rgb", "spectrum"]:
            if len(tokens) >= 3:
                return tuple(float(tok) for tok in tokens[:3])
            else:
                return None
        elif param_type == "normal":
            return tuple(float(tok) for tok in tokens)
        elif param_type == "bool":
            return tokens[0].lower() == "true"
        elif param_type in ["string", "texture"]:
            return tokens[0]
        else:
            return cleaned
    except Exception:
        return None

def to_vec3(val):
    """
    Convert an iterable of numbers to a Taichi 3D vector.

    Args:
        val (iterable): An iterable containing three numeric values.

    Returns:
        ti.Vector: A Taichi vector constructed from the given values, or a zero vector on failure.
    """
    try:
        return ti.Vector([float(x) for x in val])
    except Exception:
        return ti.Vector([0.0, 0.0, 0.0])

def py_cross(a, b):
    """
    Compute the cross product of two vectors using NumPy.

    Args:
        a (iterable): The first vector.
        b (iterable): The second vector.

    Returns:
        numpy.ndarray: The cross product of the two vectors.
    """
    return np.cross(np.array(a), np.array(b))

def py_normalize(v):
    """
    Normalize a vector using NumPy and return a Taichi vec3.

    Args:
        v (iterable): The vector to normalize.

    Returns:
        vec3: A normalized 3D vector. If the input vector has zero length, returns a zero vector.
    """
    arr = np.array(v, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return vec3(arr[0], arr[1], arr[2])

def set_matrix(m):
    """
    Validate and set a 4x4 transformation matrix. If the provided matrix is invalid, return the identity matrix.

    Args:
        m (iterable): An iterable of 16 numeric values representing a 4x4 matrix.

    Returns:
        tuple: A tuple representing the valid 4x4 matrix.
    """
    IDENTITY_4x4 = (1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0)
    return tuple(m) if m and len(m) == 16 else IDENTITY_4x4

def multiply_matrix4(A, B):
    """
    Multiply two 4x4 matrices A and B.

    Args:
        A (iterable): The first 4x4 matrix as an iterable of 16 numbers.
        B (iterable): The second 4x4 matrix as an iterable of 16 numbers.

    Returns:
        tuple: A tuple representing the resulting 4x4 matrix after multiplication.
    """
    C = [0.0]*16
    for r in range(4):
        for c in range(4):
            val = 0.0
            for k in range(4):
                val += A[r*4+k]*B[k*4+c]
            C[r*4+c] = val
    return tuple(C)

def lookat_matrix(eye, target, up):
    """
    Create a look-at transformation matrix for a camera.

    Args:
        eye (iterable): The camera position as a 3-element iterable.
        target (iterable): The target point the camera is looking at.
        up (iterable): The up direction vector.

    Returns:
        tuple: A tuple representing the 4x4 look-at transformation matrix.
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    M = np.array([
        [right[0], true_up[0], -forward[0], eye[0]],
        [right[1], true_up[1], -forward[1], eye[1]],
        [right[2], true_up[2], -forward[2], eye[2]],
        [0.0,      0.0,        0.0,        1.0]
    ], dtype=float)
    return tuple(M.flatten())
