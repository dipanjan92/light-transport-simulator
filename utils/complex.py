import taichi as ti
from taichi.math import vec3, vec2, sqrt


@ti.func
def complex_sqr(z: vec2) -> vec2:

    return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)

@ti.func
def complex_mul(a: vec2, b: vec2) -> vec2:
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def complex_add(a: vec2, b: vec2) -> vec2:
    return a + b

@ti.func
def complex_sub(a: vec2, b: vec2) -> vec2:
    return a - b

@ti.func
def complex_conjugate(z: vec2) -> vec2:
    return vec2(z.x, -z.y)

@ti.func
def complex_div(a: vec2, b: vec2) -> vec2:
    denom = b.x * b.x + b.y * b.y
    ret = vec2(0.0, 0.0)
    if denom != 0.0:
        ret = complex_mul(a, complex_conjugate(b)) / denom
    return ret

@ti.func
def complex_norm(z: vec2) -> ti.f32:
    # Returns the squared magnitude
    return z.x * z.x + z.y * z.y

@ti.func
def complex_sqrt(z: vec2) -> vec2:
    # Computes the principal square root of a complex number.
    mag = sqrt(z.x * z.x + z.y * z.y)
    real_part = sqrt(0.5 * (mag + z.x))
    imag_part = sqrt(0.5 * (mag - z.x))
    if z.y < 0.0:
        imag_part = -imag_part
    return vec2(real_part, imag_part)