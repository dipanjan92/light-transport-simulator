"""
This module implements complex arithmetic utility functions for light transport simulation.
It provides operations on complex numbers represented as 2D vectors (vec2), including multiplication,
division, conjugation, and computing the square, norm, and square root.
"""

import taichi as ti
from taichi.math import vec3, vec2, sqrt


@ti.func
def complex_sqr(z: vec2) -> vec2:
    """Return the complex square of z (as a vec2) using the formula: (x^2 - y^2, 2xy)."""
    return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)

@ti.func
def complex_mul(a: vec2, b: vec2) -> vec2:
    """Return the product of two complex numbers a and b, represented as vec2, using standard complex multiplication."""
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def complex_add(a: vec2, b: vec2) -> vec2:
    """Return the sum of two complex numbers a and b (element-wise addition)."""
    return a + b

@ti.func
def complex_sub(a: vec2, b: vec2) -> vec2:
    """Return the difference between two complex numbers a and b (element-wise subtraction)."""
    return a - b

@ti.func
def complex_conjugate(z: vec2) -> vec2:
    """Return the complex conjugate of z (i.e., flip the sign of the imaginary part)."""
    return vec2(z.x, -z.y)

@ti.func
def complex_div(a: vec2, b: vec2) -> vec2:
    """Return the division of complex number a by b, handling division by zero by returning (0,0) if necessary."""
    denom = b.x * b.x + b.y * b.y
    ret = vec2(0.0, 0.0)
    if denom != 0.0:
        ret = complex_mul(a, complex_conjugate(b)) / denom
    return ret

@ti.func
def complex_norm(z: vec2) -> ti.f32:
    """Return the squared magnitude (norm) of the complex number z."""
    return z.x * z.x + z.y * z.y

@ti.func
def complex_sqrt(z: vec2) -> vec2:
    """Return the principal square root of the complex number z using the standard formula."""
    mag = sqrt(z.x * z.x + z.y * z.y)
    real_part = sqrt(0.5 * (mag + z.x))
    imag_part = sqrt(0.5 * (mag - z.x))
    if z.y < 0.0:
        imag_part = -imag_part
    return vec2(real_part, imag_part)