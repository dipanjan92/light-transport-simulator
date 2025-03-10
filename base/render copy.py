import taichi as ti
from taichi.math import vec3
from base.lights import UniformLightSampler
from integrators.mis_pt import trace_mis
from integrators.path_trace import path_trace
from primitives.ray import Ray
from samplers.hash import hash_pixel


@ti.kernel
def render(scene: ti.template(), image: ti.template(), lights: ti.template(), camera: ti.template(),
           primitives: ti.template(), bvh: ti.template(), base_sobol_sampler: ti.template()):
    light_sampler = UniformLightSampler(lights.shape[0])
    height = camera.height
    width = camera.width
    samples_per_pixel = scene.spp
    scale_value = int(ti.max(width, height))

    ti.loop_config(parallelize=8, block_dim=32)  # Adjust for optimal performance
    for j, i in ti.ndrange(height, width):
        # Generate a unique seed for this pixel
        pixel_seed = hash_pixel(ti.Vector([i, j]), base_sobol_sampler.seed[None])

        # Clone the sampler with a new seed
        pixel_sampler = base_sobol_sampler.clone()
        pixel_sampler.seed[None] = pixel_seed

        L = vec3(0.0)
        for k in range(samples_per_pixel):
            # Proper initialization per sample
            pixel_sampler.start_pixel_sample(ti.Vector([i, j]), k, 0)

            # Generate the camera samples after initializing the sampler
            u_offset = pixel_sampler.get_1d()
            lens_sample = pixel_sampler.get_2d()

            u = (i + u_offset) / width
            v = 1.0 - (j + lens_sample[0]) / height

            # Correctly generate ray with valid sampler input
            ray_origin, ray_direction = camera.generate_ray(u, v, pixel_sampler)
            ray = Ray(ray_origin, ray_direction)

            # Choose integrator explicitly to avoid ambiguity
            if scene.integrator == 0:
                L += path_trace(ray, primitives, bvh, lights, light_sampler, pixel_sampler,
                               sample_lights=scene.sample_lights,
                               sample_bsdf=scene.sample_bsdf,
                               max_depth=scene.max_depth)
            else:
                L += trace_mis(ray, primitives, bvh, lights, light_sampler, pixel_sampler,
                               sample_lights=scene.sample_lights,
                               sample_bsdf=scene.sample_bsdf,
                               max_depth=scene.max_depth)

        image[j, i] = L / samples_per_pixel
