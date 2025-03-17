import taichi as ti
from taichi.math import vec3, vec2, dot, max, isinf

from accelerators.bvh import intersect_bvh, unoccluded
from base.bsdf import BXDF_SPECULAR, BXDF_REFLECTION, BXDF_TRANSMISSION, BXDF_NONE, IMPORTANCE
from base.lights import uniform_sample_one_light, is_black, power_heuristic
from primitives.intersects import Intersection
from primitives.ray import Ray, spawn_ray, offset_ray_origin
from utils.constants import INF


@ti.func
def trace_mis(ray, primitives, bvh, lights, light_sampler, sample_lights=1, sample_bsdf=1, max_depth=3):
    """
    Performs path tracing using Multiple Importance Sampling (MIS).

    Args:
        ray: The initial ray to trace.
        primitives: Scene primitives.
        bvh: Bounding Volume Hierarchy for acceleration.
        lights: List of light sources.
        light_sampler: Sampler for selecting lights.
        sample_lights: Flag to determine whether to sample direct illumination from lights (default: 1 for True).
        sample_bsdf: Flag to determine whether to sample the BSDF (default: 1 for True).
        max_depth: Maximum recursion depth (default: 3).

    Returns:
        vec3: The computed radiance along the ray.
    """
    # Main path tracing loop: intersect the scene, accumulate radiance,
    # sample BSDF, and apply Russian roulette until termination.
    
    L = vec3(0.0)
    beta = vec3(1.0)  # Path throughput
    depth = 0  # Depth of the recursion
    eta_scale = 1.0  # IoR scale
    specular_bounce = 0
    any_non_specular_bounces = 0
    regularize = 0
    p_b = 1.0  # Probability density of the previous bounce
    prev_intr_ctx = Intersection()  # Previous interaction context for MIS
    t_max = INF
    t_min = 0.0

    while 1:
        # Intersect scene
        isect = intersect_bvh(ray, primitives, bvh, t_min, t_max)
        if not isect.intersected:
            break

        # Check if emissive
        Le = vec3(0.0)
        Le += isect.Le(-ray.direction)

        if not is_black(Le):
            if depth == 0 or specular_bounce:
                L += beta * Le
            else:
                # Compute MIS weight for area light
                if isect.nearest_object.is_light:
                    # should be true always
                    area_light = lights[isect.nearest_object.light_idx]
                    p_l = light_sampler.pmf() * area_light.pdf_Li(prev_intr_ctx, ray.direction, isect.nearest_object.triangle)
                    w_l = power_heuristic(1, p_b, 1, p_l)
                    L += beta * w_l * Le

        # Get BSDF
        bsdf = isect.get_bsdf()

        if bsdf.flags() == BXDF_NONE:
            specular_bounce = True
            ray = spawn_ray(isect.intersected_point, isect.normal, ray.direction)
            continue

        # Regularize the BSDF if necessary
        if regularize and any_non_specular_bounces:
            print("need to implement base regularize method")
            # bsdf.regularize()

        # End path if maximum depth is reached
        if depth == max_depth:
            break
        depth += 1

        # Sample direct illumination
        if bsdf.flags() & BXDF_SPECULAR == 0:
            Ld = sample_Ld(ray, primitives, bvh, isect, bsdf, light_sampler, lights, rng)
            L += beta * Ld

        # u = 0.0
        # u2 = vec2(0.0)

        # # Sample BSDF (QMC for specular, RNG for diffuse)
        # if bsdf.flags() & BXDF_SPECULAR != 0:
        #     u = sampler.get_1d()
        #     u2 = sampler.get_2d()  # Stratified
        # else:
        u = rng.get_1d()
        u2 = rng.get_2d()     # Stochastic

        # Sample BSDF
        wo = -ray.direction
        bs = bsdf.sample_f(wo, u, u2)

        if is_black(bs.f) or bs.pdf == 0:
            break

        # surface scattering
        beta *= bs.f * ti.abs(dot(bs.wi, isect.normal)) / bs.pdf
        p_b = bs.pdf # this would work for everything but LayeredBXDF (not implemented)
        specular_bounce = (bs.flags & BXDF_SPECULAR != 0)
        any_non_specular_bounces |= bs.flags & BXDF_SPECULAR == 0
        if bs.flags & BXDF_TRANSMISSION != 0:
            eta_scale *= bs.eta**2

        prev_intr_ctx = isect

        ray = spawn_ray(isect.intersected_point, isect.normal, bs.wi)


        # Russian roulette
        rr_beta = beta * eta_scale
        if rr_beta.max() < 1.0 and depth > 5:
            q = max(0.0, 1.0 - rr_beta.max())
            if rng.get_1d() < q:
                break
            beta /= (1.0 - q)

    return L


@ti.func
def sample_Ld(ray, primitives, bvh, isect, bsdf, light_sampler, lights):
    """
    Samples direct illumination for a given intersection.

    Args:
        ray: The incident ray.
        primitives: Scene primitives.
        bvh: Bounding Volume Hierarchy for acceleration.
        isect: Intersection data at the current point.
        bsdf: Material BSDF at the intersection.
        light_sampler: Sampler for selecting lights.
        lights: List of light sources.

    Returns:
        vec3: The computed direct lighting contribution.
    """
    # Adjust the light sampling position based on the BSDF properties (reflection or transmission)
    Ld = vec3(0.0)

    # Initialize LightSampleContext for light sampling
    ctx_p = vec3(0.0)

    # Adjust the light sampling position based on BSDF flags
    if bsdf.flags() & BXDF_REFLECTION != 0 and bsdf.flags() & BXDF_TRANSMISSION == 0:
        ctx_p = offset_ray_origin(isect.intersected_point, isect.normal, -ray.direction)
    elif bsdf.flags() & BXDF_TRANSMISSION != 0 and bsdf.flags() & BXDF_REFLECTION == 0:
        ctx_p = offset_ray_origin(isect.intersected_point, isect.normal, ray.direction)

    # Choose a light source for direct lighting calculation
    s_l = light_sampler.sample(rng.get_1d())
    sampled_li = lights[s_l.light_idx]
    u_light = rng.get_2d()
    # Light sampling (TODO)
    # if light.is_area:
    #     u_light = sampler.get_2d()  # QMC
    # else:
    #     u_light = rng.get_2d()      # RNG
    l_shape = primitives[sampled_li.shape_idx].triangle
    ls = sampled_li.sample_Li(ctx_p, u_light, l_shape)

    if not is_black(ls.L) and ls.pdf > 0:
        wi = ls.wi
        f = bsdf.f(-ray.direction, wi, IMPORTANCE) * ti.abs(dot(wi, isect.normal))
        if not is_black(f) and unoccluded(ctx_p, isect.normal, ls.intr_p, primitives, bvh, 1e-4):
            p_l = s_l.pdf * ls.pdf

            # Delta light not implemented

            p_b = bsdf.pdf(-ray.direction, wi)
            w_l = power_heuristic(1, p_l, 1, p_b)
            Ld = w_l * ls.L * f / p_l

    return Ld
