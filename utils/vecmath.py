"""
This module implements various vector math utility functions for light transport simulation.
It provides functions for computing trigonometric, geometric, and arithmetic operations on vectors,
including safe square roots, dot products, cross products, interpolation, and more.
"""

import taichi as ti
from taichi.math import vec3, sqrt, vec2, atan2, cross, dot, length

from utils.constants import *


@ti.func
def same_hemisphere(wo: vec3, wi: vec3) -> bool:
    """Return True if both vectors are in the same hemisphere (i.e., have the same sign in z)."""
    
    ret = False
    if (wo.z * wi.z) > 0.0:
        ret = True
    return ret


@ti.func
def cos_theta(w: vec3) -> ti.f32:
    """Return the cosine of the angle with respect to the z-axis (assumes w.z is cosine)."""
    
    ret = w.z
    return ret


@ti.func
def abs_cos_theta(w: vec3) -> ti.f32:
    """Return the absolute value of the cosine of the angle (i.e., |w.z|)."""
    
    ret = ti.abs(w.z)
    return ret


@ti.func
def sqr(x: ti.f32) -> ti.f32:
    """Return the square of x."""
    
    ret = x * x
    return ret


@ti.func
def is_inf(x: ti.f32) -> bool:
    """Return True if x is considered infinite (absolute value > 1e30)."""
    
    ret = False
    if ti.abs(x) > 1e30:
        ret = True
    return ret


@ti.func
def tan2_theta(w: vec3) -> ti.f32:
    """Return the square of the tangent of the angle from the z-axis; if cos^2 is zero, return a large value (1e30)."""

    cos2 = w.z * w.z
    sin2 = ti.max(0.0, 1.0 - cos2)
    ret = 0.0
    if cos2 == 0.0:
        # infinite
        ret = 1e30
    else:
        ret = sin2 / cos2
    return ret


@ti.func
def cos2_theta(w: vec3) -> ti.f32:
    """Return the square of the cosine of the angle (i.e., w.z squared)."""
    
    ret = w.z * w.z
    return ret


@ti.func
def cos_phi(w: vec3) -> ti.f32:
    """Return the cosine of the azimuthal angle computed from x and y components."""
    
    denom = w.x * w.x + w.y * w.y
    ret = 0.0
    if denom > 0.0:
        ret = w.x / sqrt(denom)
    return ret


@ti.func
def sin_phi(w: vec3) -> ti.f32:
    """Return the sine of the azimuthal angle computed from x and y components."""
    
    denom = w.x * w.x + w.y * w.y
    ret = 0.0
    if denom > 0.0:
        ret = w.y / sqrt(denom)
    return ret


@ti.func
def abs_dot(a: vec3, b: vec3) -> ti.f32:
    """Return the absolute value of the dot product of vectors a and b."""
    
    dot_val = a.x * b.x + a.y * b.y + a.z * b.z
    ret = ti.abs(dot_val)
    return ret


@ti.func
def lerp(t: ti.f32, v1: ti.f32, v2: ti.f32) -> ti.f32:
    """Return the linear interpolation between v1 and v2 with parameter t."""
    
    ret = (1.0 - t) * v1 + t * v2
    return ret


@ti.func
def safe_sqrt(x):
    """Return the square root of x, ensuring non-negative input by clamping x to at least 0.0."""
    return sqrt(max(0.0, x))


@ti.func
def length_squared(v):
    """Return the squared length of vector v."""
    return length(v) ** 2


@ti.func
def distance_squared(u, v):
    """Return the squared distance between vectors u and v."""
    return length_squared(u - v)


@ti.func
def spherical_triangle_area(a, b, c):
    """Return the area of a spherical triangle defined by vertices a, b, and c using the atan2 formula."""
    return ti.abs(2 * atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)))


@ti.func
def max_component(vec):
    """Return the maximum component value of the vector."""
    max_val = vec[0]
    for i in range(1, vec.n):
        max_val = max(max_val, vec[i])
    return max_val


@ti.func
def gamma(n):
    """Return the gamma correction term for n samples, using a machine epsilon of 1e-7."""
    machine_epsilon = ti.f32(1e-7)
    return (n * machine_epsilon) / (1 - n * machine_epsilon)


@ti.func
def face_forward(v, n):
    """Return v with its direction flipped if it is opposite to n (i.e., if dot(v, n) < 0)."""
    return -v if dot(v, n) < 0 else v


@ti.func
def safe_inverse(x):
    """Return the inverse of x safely; if |x| is too small (less than EPSILON), use 1/EPSILON instead."""
    return 1.0 / x if abs(x) > EPSILON else 1.0 / EPSILON


@ti.func
def max_component(vec):
    """Return the maximum component value of the vector."""
    max_val = vec[0]
    for i in range(1, vec.n):
        max_val = max(max_val, vec[i])
    return max_val