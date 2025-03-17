import taichi as ti
from taichi.math import vec3, sqrt, vec2, atan2, cross, dot, length

from utils.constants import *


@ti.func
def same_hemisphere(wo: vec3, wi: vec3) -> bool:
    
    ret = False
    if (wo.z * wi.z) > 0.0:
        ret = True
    return ret


@ti.func
def cos_theta(w: vec3) -> ti.f32:
    
    ret = w.z
    return ret


@ti.func
def abs_cos_theta(w: vec3) -> ti.f32:
    
    ret = ti.abs(w.z)
    return ret


@ti.func
def sqr(x: ti.f32) -> ti.f32:
    
    ret = x * x
    return ret


@ti.func
def is_inf(x: ti.f32) -> bool:
    ret = False
    if ti.abs(x) > 1e30:
        ret = True
    return ret


@ti.func
def tan2_theta(w: vec3) -> ti.f32:

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
    
    ret = w.z * w.z
    return ret


@ti.func
def cos_phi(w: vec3) -> ti.f32:

    denom = w.x * w.x + w.y * w.y
    ret = 0.0
    if denom > 0.0:
        ret = w.x / sqrt(denom)
    return ret


@ti.func
def sin_phi(w: vec3) -> ti.f32:

    denom = w.x * w.x + w.y * w.y
    ret = 0.0
    if denom > 0.0:
        ret = w.y / sqrt(denom)
    return ret


@ti.func
def abs_dot(a: vec3, b: vec3) -> ti.f32:
    
    dot_val = a.x * b.x + a.y * b.y + a.z * b.z
    ret = ti.abs(dot_val)
    return ret


@ti.func
def lerp(t: ti.f32, v1: ti.f32, v2: ti.f32) -> ti.f32:
    
    ret = (1.0 - t) * v1 + t * v2
    return ret


@ti.func
def safe_sqrt(x):
    return sqrt(max(0.0, x))


@ti.func
def length_squared(v):
    return length(v) ** 2


@ti.func
def distance_squared(u, v):
    return length_squared(u - v)


@ti.func
def spherical_triangle_area(a, b, c):
    return ti.abs(2 * atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)))


@ti.func
def max_component(vec):
    max_val = vec[0]
    for i in range(1, vec.n):
        max_val = max(max_val, vec[i])
    return max_val


@ti.func
def gamma(n):
    machine_epsilon = ti.f32(1e-7)
    return (n * machine_epsilon) / (1 - n * machine_epsilon)


@ti.func
def face_forward(v, n):
    return -v if dot(v, n) < 0 else v


@ti.func
def safe_inverse(x):
    return 1.0 / x if abs(x) > EPSILON else 1.0 / EPSILON


@ti.func
def max_component(vec):
    max_val = vec[0]
    for i in range(1, vec.n):
        max_val = max(max_val, vec[i])
    return max_val