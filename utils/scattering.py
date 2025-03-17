import taichi as ti
from taichi.math import vec3, vec2, cos, sin, sqrt, dot, max, clamp
from utils.complex import *
from utils.constants import *


@ti.func
def sample_uniform_disk_polar(u: vec2) -> vec2:

    r = sqrt(u[0])
    theta = 2.0 * PI * u[1]
    px = r * cos(theta)
    py = r * sin(theta)
    ret = vec2(px, py)
    return ret


@ti.func
def reflect(wo: vec3, n: vec3) -> vec3:
    dot_val = dot(wo, n)
    return -wo + 2.0 * dot_val * n


@ti.func
def refract(wi, n, eta):

    cosTheta_i = dot(n, wi)
    local_eta = eta
    local_n = n

    # Potentially flip interface orientation for Snell's law if cosTheta_i < 0.
    if cosTheta_i < 0.0:
        local_eta = 1.0 / local_eta
        cosTheta_i = -cosTheta_i
        local_n = -local_n

    sin2Theta_i = 1.0 - cosTheta_i * cosTheta_i
    sin2Theta_i = max(sin2Theta_i, 0.0)

    sin2Theta_t = sin2Theta_i / (local_eta * local_eta)

    valid = 1
    wt = vec3(0.0, 0.0, 0.0)
    ret_etap = local_eta

    # Check for total internal reflection.
    if sin2Theta_t >= 1.0:
        valid = 0
    else:
        cosTheta_t = ti.sqrt(1.0 - sin2Theta_t)

        inv_eta = 1.0 / local_eta
        wt = (-wi * inv_eta) + (cosTheta_i * inv_eta - cosTheta_t) * local_n
        wt = wt.normalized()

    return valid, wt, ret_etap


@ti.func
def face_forward(n: vec3, n2: vec3) -> vec3:

    ret = n
    if dot(n, n2) < 0.0:
        ret = -n
    return ret


@ti.func
def fresnel(cos_theta_i, eta):
    # Initialize output variables
    r = 0.0
    cos_theta_t = 0.0
    eta_it = 0.0
    eta_ti = 0.0

    # Check if the ray is entering or exiting the surface
    outside_mask = cos_theta_i >= 0.0

    rcp_eta = 1.0 / eta
    eta_it = eta if outside_mask else rcp_eta
    eta_ti = rcp_eta if outside_mask else eta

    # Calculate the squared sine of the transmitted angle using Snell's law
    sin2_theta_t = eta_ti * eta_ti * (1.0 - cos_theta_i * cos_theta_i)
    cos_theta_t_sqr = 1.0 - sin2_theta_t

    # Absolute cosines of the incident and transmitted rays
    cos_theta_i_abs = ti.abs(cos_theta_i)
    cos_theta_t_abs = ti.sqrt(ti.max(0.0, cos_theta_t_sqr))

    # Handle special cases where the index is matched or cos_theta_i is zero
    index_matched = eta == 1.0
    special_case = index_matched or (cos_theta_i_abs == 0.0)

    r_sc = 0.0 if index_matched else 1.0

    # Calculate the reflection coefficients
    a_s = (eta_it * cos_theta_t_abs - cos_theta_i_abs) / (eta_it * cos_theta_t_abs + cos_theta_i_abs)
    a_p = (eta_it * cos_theta_i_abs - cos_theta_t_abs) / (eta_it * cos_theta_i_abs + cos_theta_t_abs)

    r = 0.5 * (a_s * a_s + a_p * a_p)

    # Apply the special case handling
    if special_case:
        r = r_sc

    # Adjust the sign of the transmitted direction
    if cos_theta_i < 0:
        cos_theta_t = -cos_theta_t_abs
    else:
        cos_theta_t = cos_theta_t_abs

    return r, cos_theta_t, eta_it, eta_ti


@ti.func
def fr_dielectric(cos_theta_i: ti.f32, eta: ti.f32) -> ti.f32:
    # Clamp cosTheta_i to [-1, 1].
    c = clamp(cos_theta_i, -1.0, 1.0)
    ret = 0.0  # Final Fresnel reflectance.

    # Potentially flip interface orientation if cosTheta_i < 0.
    local_eta = eta
    local_c = c
    if c < 0.0:
        local_eta = 1.0 / eta
        local_c = -c

    sin2_i = 1.0 - (local_c * local_c)
    sin2_t = sin2_i / (local_eta * local_eta)

    # Check total internal reflection.
    if sin2_t >= 1.0:
        ret = 1.0
    else:
        # Compute cosTheta_t = sqrt(1 - sin2_t).
        cos_theta_t = ti.sqrt(1.0 - sin2_t)

        # Fresnel reflection for parallel and perpendicular polarizations.
        r_parl = (local_eta * local_c - cos_theta_t) / (local_eta * local_c + cos_theta_t)
        r_perp = (local_c - local_eta * cos_theta_t) / (local_c + local_eta * cos_theta_t)

        # Final reflectance is the average of squared magnitudes.
        ret = 0.5 * (r_parl * r_parl + r_perp * r_perp)

    return ret

@ti.func
def fr_complex_conductor(cosTheta_i: ti.f32, eta: vec2) -> ti.f32:

    c = clamp(cosTheta_i, 0.0, 1.0)

    sin2Theta_i = 1.0 - c * c

    eta_sq = complex_sqr(eta)

    sin2Theta_i_complex = vec2(sin2Theta_i, 0.0)
    sin2Theta_t = complex_div(sin2Theta_i_complex, eta_sq)

    one_complex = vec2(1.0, 0.0)
    sub_val = complex_sub(one_complex, sin2Theta_t)
    cosTheta_t = complex_sqrt(sub_val)

    c_complex = vec2(c, 0.0)
    eta_times_c = complex_mul(eta, c_complex)
    num_parl = complex_sub(eta_times_c, cosTheta_t)
    den_parl = complex_add(eta_times_c, cosTheta_t)
    r_parl = complex_div(num_parl, den_parl)

    eta_cosTheta_t = complex_mul(eta, cosTheta_t)
    num_perp = complex_sub(c_complex, eta_cosTheta_t)
    den_perp = complex_add(c_complex, eta_cosTheta_t)
    r_perp = complex_div(num_perp, den_perp)

    result = (complex_norm(r_parl) + complex_norm(r_perp)) / 2.0
    return result

@ti.func
def fr_complex(cosTheta_i: ti.f32, eta: vec3, k: vec3) -> vec3:

    ret = vec3(0.0)
    for i in ti.static(range(3)):
        ret[i] = fr_complex_conductor(cosTheta_i, vec2(eta[i], k[i]))
    return ret


@ti.func
def sample_cosine_hemisphere(u):
    
    r = sqrt(u[0])
    theta = 2.0 * PI * u[1]
    x = r * cos(theta)
    y = r * sin(theta)
    z = sqrt(ti.max(0.0, 1.0 - x*x - y*y))
    ret = vec3(x, y, z)
    return ret


@ti.func
def cosine_hemisphere_pdf(cos_t):
    
    ret = 0.0
    if cos_t > 0.0:
        ret = cos_t * INV_PI
    return ret