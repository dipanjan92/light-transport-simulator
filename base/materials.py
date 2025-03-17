import taichi as ti
from taichi.math import pi, vec3, length

"""
Module for material definitions in the light transport simulator.

This module defines the Material class, which encapsulates the physical properties of a material such as reflectance, transmittance, roughness, refractive index (eta), extinction coefficient (k), and emission. These properties are used during light transport and rendering computations.
"""

from base.bsdf import BSDF


@ti.dataclass
class Material:
    """
    Material properties for rendering.

    Attributes:
        material_type (ti.i32): Identifier for the type of material.
        reflectance (vec3): Reflective color of the material.
        transmittance (vec3): Transmissive color of the material.
        uroughness (ti.f32): Roughness along the 'u' (tangent) direction.
        vroughness (ti.f32): Roughness along the 'v' (bitangent) direction.
        eta (vec3): Refractive index of the material.
        k (vec3): Extinction coefficient of the material.
        emission (vec3): Emissive color of the material.
        edited (ti.i32): Flag indicating whether the material has been modified.
    """
    material_type: ti.i32
    reflectance: vec3
    transmittance: vec3
    uroughness: ti.f32   # Roughness along the "u" (tangent) direction.
    vroughness: ti.f32   # Roughness along the "v" (bitangent) direction.
    eta: vec3
    k: vec3
    emission: vec3
    edited: ti.i32