"""
This module implements the scene configuration for light transport simulation.
It defines a Scene class that encapsulates the integrator type, samples per pixel, maximum recursion depth,
and sampling parameters for lights and BSDF.
"""

import taichi as ti
from taichi.math import vec3, normalize, cross, sin, cos

from base.camera import PerspectiveCamera


@ti.dataclass
class Scene:
    """
    Represents the configuration for a scene in light transport simulation.
    Contains parameters such as integrator type, samples per pixel (spp), maximum ray recursion depth,
    and flags for sampling lights and BSDF components.
    """
    integrator: ti.i32
    spp: ti.i32
    max_depth: ti.i32
    sample_lights: ti.i32
    sample_bsdf: ti.i32