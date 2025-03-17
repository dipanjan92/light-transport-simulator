"""
This module implements a path tracing integrator for light transport simulation.
It calculates radiance along a ray by recursively tracing paths through the scene,
sampling BSDF and direct lighting contributions, and incorporating Russian roulette termination.
"""

import taichi as ti
from taichi.math import vec3, vec2, dot, max, isinf

from accelerators.bvh import intersect_bvh, unoccluded
from base.bsdf import BXDF_SPECULAR, BXDF_REFLECTION, BXDF_TRANSMISSION, BXDF_NONE
from base.lights import uniform_sample_one_light, is_black
from base.samplers import sample_uniform_sphere, uniform_sphere_pdf, sample_uniform_hemisphere, uniform_hemisphere_pdf
from primitives.ray import Ray, spawn_ray
from utils.constants import INF


@ti.func
def path_trace(ray, primitives, bvh, lights, light_sampler, sample_lights=1, sample_bsdf=1, max_depth=3):
    """
    Performs path tracing for light transport simulation.

    Args:
        ray: The initial ray to trace.
        primitives: Scene primitives.
        bvh: Bounding Volume Hierarchy used for acceleration.
        lights: List of light sources.
        light_sampler: Sampler for selecting lights.
        sample_lights: Flag to determine whether to sample direct illumination from lights (default: 1 for True).
        sample_bsdf: Flag to determine whether to sample the BSDF (default: 1 for True).
        max_depth: Maximum recursion depth (default: 3).

    Returns:
        vec3: The accumulated radiance along the ray.
    """
    L = vec3(0.0)  # Initialize accumulated radiance
    beta = vec3(1.0)  # Path throughput (multiplicative factor for radiance)
    specular_bounce = 1  # Flag to indicate whether the previous bounce was specular
    depth = 0  # Depth of the recursion
    t_max = INF  # Maximum distance for intersection
    t_min = 0.0  # Minimum distance for intersection

    while 1:
        if depth >= max_depth:
            break

        # Intersect the ray with the scene using the BVH
        isect = intersect_bvh(ray, primitives, bvh, t_min, t_max)

        if not isect.intersected:
            # If no intersection, terminate. TODO: Add environment lighting if available.
            break

        # Accumulate emitted light from the intersected object
        L += beta * isect.nearest_object.material.emission

        # Russian roulette: if depth > 4, probabilistically terminate path based on reflectance
        if depth > 4:
            r_r = isect.nearest_object.material.reflectance.max()
            if ti.random() >= r_r:
                break
            beta = beta/r_r

        depth += 1

        # Get the BSDF of the intersected object to determine scattering behavior
        bsdf = isect.get_bsdf()

        wo = -ray.direction  # Outgoing direction from the intersection point

        # Direct lighting contribution
        if sample_lights:
            # Sample a light source using the provided sampler
            s_l = light_sampler.sample(ti.random())
            sampled_li = lights[s_l.light_idx]
            u_light = vec2(ti.random(), ti.random())
            l_shape = primitives[sampled_li.shape_idx].triangle
            ls = sampled_li.sample_Li(isect.intersected_point, u_light, l_shape)

            # If light sample is valid, compute contribution
            if not is_black(ls.L) and ls.pdf > 0:
                # Incoming light direction from the light source
                wi = ls.wi
                f = bsdf.f(wo, wi) * ti.abs(dot(wi, isect.normal))
                if not is_black(f) and unoccluded(isect.intersected_point, isect.normal, ls.intr_p, primitives, bvh, 1e-4):
                    L += beta * (f * ls.L / ls.pdf) / s_l.pdf

        # BSDF sampling: sample the BSDF to determine new ray direction
        u = ti.random()
        u2 = vec2(ti.random(), ti.random())
        bs = bsdf.sample_f(wo, u, u2)

        # Terminate if BSDF sample is invalid
        if is_black(bs.f) or bs.pdf == 0:
            break

        # Update the path throughput with the BSDF sample
        beta *= bs.f * ti.abs(dot(bs.wi, isect.normal)) / bs.pdf
        specular_bounce = (bs.flags & BXDF_SPECULAR != 0)
        wi = bs.wi
        # Generate a new ray from the intersection point in the sampled direction
        ray = spawn_ray(isect.intersected_point, isect.normal, wi)

    return L
