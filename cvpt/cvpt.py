import taichi as ti
from taichi.math import vec3, vec2, dot, max
from accelerators.bvh import intersect_bvh, unoccluded
from base.bsdf import BXDF_SPECULAR, BXDF_REFLECTION, BXDF_TRANSMISSION, BXDF_NONE, IMPORTANCE
from base.lights import uniform_sample_one_light, is_black, power_heuristic
from primitives.intersects import Intersection
from primitives.ray import Ray, spawn_ray, offset_ray_origin
from utils.constants import INF

@ti.func
def trace_cvpt(ray, primitives, bvh, lights, light_sampler, updated_bsdf, sample_lights=1, sample_bsdf=1, max_depth=3):
    """
    Performs cross-material path tracing for dual-material light transport simulation.

    Args:
        ray: The initial ray to trace.
        primitives: Scene primitives.
        bvh: Bounding Volume Hierarchy for acceleration.
        lights: List of light sources.
        light_sampler: Sampler for selecting lights.
        updated_bsdf: Updated BSDF for the edited material.
        sample_lights: Flag to determine whether to sample direct illumination from lights (default: 1 for True).
        sample_bsdf: Flag to determine whether to sample the BSDF (default: 1 for True).
        max_depth: Maximum recursion depth (default: 3).

    Returns:
        (vec3, vec3): Tuple of radiance estimates for the new and old materials respectively.
    """
    # Initialize separate accumulators for new and old estimates
    L_new = vec3(0.0)
    L_old = vec3(0.0)
    beta_new = vec3(1.0)
    beta_old = vec3(1.0)
    depth = 0
    eta_scale = 1.0
    specular_bounce = 0
    any_non_specular_bounces = 0
    p_b = 1.0
    prev_intr_ctx = Intersection()
    t_max = INF
    t_min = 0.0

    while 1:
        # Intersect scene using new geometry (assumed to be same for both materials)
        isect = intersect_bvh(ray, primitives, bvh, t_min, t_max)
        if not isect.intersected:
            break

        # Evaluate emission:
        Le = vec3(0.0)
        Le += isect.Le(-ray.direction)
        if not is_black(Le):
            if depth == 0 or specular_bounce:
                L_new += beta_new * Le
                L_old += beta_old * Le
            else:
                if isect.nearest_object.is_light:
                    area_light = lights[isect.nearest_object.light_idx]
                    p_l = light_sampler.pmf() * area_light.pdf_Li(prev_intr_ctx, ray.direction, isect.nearest_object.triangle)
                    w_l = power_heuristic(1, p_b, 1, p_l)
                    L_new += beta_new * w_l * Le
                    L_old += beta_old * w_l * Le

        # Get BSDF for new and old materials.
        # These functions should be provided by your dual-material type.
        bsdf_new = isect.get_bsdf()
        bsdf_old = isect.get_bsdf()

        # Check if the material is edited
        if isect.nearest_object.material.edited:
            updated_bsdf.frame = isect.frame
            bsdf_new = updated_bsdf

        # If no valid BSDF, continue as a specular bounce.
        if bsdf_new.flags() == BXDF_NONE:
            specular_bounce = True
            ray = spawn_ray(isect.intersected_point, isect.normal, ray.direction)
            continue

        if depth == max_depth:
            break
        depth += 1

        # Sample direct illumination if not a specular bounce
        if bsdf_new.flags() & BXDF_SPECULAR == 0:
            randoms = vec3([ti.random(), ti.random(), ti.random()])
            Ld_new = sample_Ld(ray, primitives, bvh, isect, bsdf_new, light_sampler, lights, randoms)
            L_new += beta_new * Ld_new
            Ld_old = sample_Ld(ray, primitives, bvh, isect, bsdf_old, light_sampler, lights, randoms)
            L_old += beta_old * Ld_old

        wo = -ray.direction
        u = ti.random()
        u2 = vec2(ti.random(), ti.random())

        # Sample BSDF for both new and old materials with the same random numbers
        bs_new = bsdf_new.sample_f(wo, u, u2)
        bs_old = bsdf_old.sample_f(wo, u, u2)

        # If either sample fails, break out of the loop
        if is_black(bs_new.f) or bs_new.pdf == 0 or is_black(bs_old.f) or bs_old.pdf == 0:
            break

        beta_new *= bs_new.f * ti.abs(dot(bs_new.wi, isect.normal)) / bs_new.pdf
        beta_old *= bs_old.f * ti.abs(dot(bs_old.wi, isect.normal)) / bs_old.pdf
        p_b = bs_new.pdf
        specular_bounce = (bs_new.flags & BXDF_SPECULAR != 0)
        any_non_specular_bounces |= (bs_new.flags & BXDF_SPECULAR == 0)
        if bs_new.flags & BXDF_TRANSMISSION != 0:
            eta_scale *= bs_new.eta**2

        prev_intr_ctx = isect
        ray = spawn_ray(isect.intersected_point, isect.normal, bs_new.wi)

        # Russian roulette termination
        rr_beta = max(beta_new.max(), beta_old.max()) * eta_scale
        if rr_beta < 1.0 and depth > 5:
            q = max(0.0, 1.0 - rr_beta)
            if ti.random() < q:
                break
            beta_new /= (1.0 - q)
            beta_old /= (1.0 - q)

    return L_new, L_old

@ti.func
def sample_Ld(ray, primitives, bvh, isect, bsdf, light_sampler, lights, randoms):
    """
    Samples direct illumination at the intersection for dual-material rendering.

    Args:
        ray: The incident ray.
        primitives: Scene primitives.
        bvh: Bounding Volume Hierarchy for acceleration.
        isect: Intersection data at the current point.
        bsdf: BSDF for the material at the intersection.
        light_sampler: Sampler for selecting lights.
        lights: List of light sources.
        randoms: A vec3 of random values used for light sampling.

    Returns:
        vec3: The computed direct lighting contribution.
    """
    Ld = vec3(0.0)

    # Initialize LightSampleContext for light sampling
    ctx_p = vec3(0.0)

    # Adjust the light sampling position based on BSDF flags
    if bsdf.flags() & BXDF_REFLECTION != 0 and bsdf.flags() & BXDF_TRANSMISSION == 0:
        ctx_p = offset_ray_origin(isect.intersected_point, isect.normal, -ray.direction)
    elif bsdf.flags() & BXDF_TRANSMISSION != 0 and bsdf.flags() & BXDF_REFLECTION == 0:
        ctx_p = offset_ray_origin(isect.intersected_point, isect.normal, ray.direction)

    # Choose a light source for direct lighting calculation
    s_l = light_sampler.sample(randoms[0])
    sampled_li = lights[s_l.light_idx]
    u_light = vec2(randoms[1], randoms[2])
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