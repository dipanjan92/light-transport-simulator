"""
This module implements the rendering kernel for light transport simulation using Taichi.
It defines a render kernel that supports multiple integrators (e.g., path tracing and MIS-based tracing)
to compute pixel colors by averaging samples per pixel.
"""
import taichi as ti
from taichi.math import vec3
from base.lights import UniformLightSampler
from integrators.mis_pt import trace_mis
from integrators.path_trace import path_trace
from primitives.ray import Ray


@ti.kernel
def render(scene: ti.template(), image: ti.template(), lights: ti.template(), camera: ti.template(),
           primitives: ti.template(), bvh: ti.template()):
    """
    Render the scene by computing pixel colors using the specified integrator.

    The kernel iterates over each pixel, generates camera rays, and accumulates radiance contributions
    from either the path tracing integrator or the MIS-based tracer, based on the scene's integrator setting.

    Args:
        scene: The scene configuration containing parameters like spp, integrator type, etc.
        image: The output image array.
        lights: Array of light sources in the scene.
        camera: The camera object used to generate rays.
        primitives: Array of scene primitives.
        bvh: The bounding volume hierarchy for the scene.
    """
    # Initialize a uniform light sampler based on the number of lights in the scene.
    light_sampler = UniformLightSampler(lights.shape[0])

    # Retrieve camera dimensions and the number of samples per pixel from the scene configuration.
    height = camera.height
    width = camera.width
    samples_per_pixel = scene.spp

    if scene.integrator == 0:
        # Using the path tracing integrator to compute pixel colors.
        ti.loop_config(parallelize=4, block_dim=16)
        for j, i in ti.ndrange(height, width):
            L = vec3(0.0)
            # For each pixel, accumulate radiance by averaging multiple samples.
            for k in range(samples_per_pixel):
                # Generate random offsets to perform subpixel sampling.
                r_u = ti.random()
                r_v = ti.random()
                # Convert pixel coordinates to normalized screen space coordinates (u, v).
                u = (i + r_u) / width
                v = 1 - (j + r_v) / height

                # Generate a camera ray based on the computed (u, v) coordinates.
                ray_origin, ray_direction = camera.generate_ray(u, v)
                ray = Ray(ray_origin, ray_direction)
                # Trace the ray using the path tracing integrator and accumulate the radiance.
                L += path_trace(ray, primitives, bvh, lights, light_sampler, 
                                sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)
            image[j, i] = L / samples_per_pixel
    else:
        # Using the MIS-based integrator for light transport simulation.
        ti.loop_config(parallelize=4, block_dim=16)
        for j, i in ti.ndrange(height, width):
            L = vec3(0.0)
            # For each pixel, accumulate radiance by averaging multiple samples.
            for k in range(samples_per_pixel):
                # Generate random offsets and convert pixel coordinates to normalized values.
                r_u = ti.random()
                r_v = ti.random()
                u = (i + r_u) / width
                v = 1 - (j + r_v) / height

                # Generate a camera ray based on these coordinates.
                ray_origin, ray_direction = camera.generate_ray(u, v)
                ray = Ray(ray_origin, ray_direction)
                # Trace the ray using the MIS-based integrator and accumulate the radiance.
                L += trace_mis(ray, primitives, bvh, lights, light_sampler,
                               sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)
            image[j, i] = L / samples_per_pixel
