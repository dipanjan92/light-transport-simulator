import taichi as ti
from taichi.math import vec3

from base.lights import UniformLightSampler
from primitives.ray import Ray
from integrators.mis_pt import trace_mis

@ti.func
def compute_variance(values: ti.template(), mean: vec3) -> vec3:
    variance = vec3(0.0)
    for k in range(values.shape[0]):
        diff = values[k] - mean
        variance += diff * diff
    return variance / values.shape[0]

@ti.kernel
def render_control_variate(scene: ti.template(),
                           image: ti.template(),
                           control: ti.template(),
                           lights: ti.template(),
                           camera: ti.template(),
                           primitives: ti.template(),
                           bvh: ti.template(),
                           cv_samples: ti.template(),
                           std_samples: ti.template()):

    light_sampler = UniformLightSampler(lights.shape[0])

    height = camera.height
    width = camera.width

    n_cv = cv_samples.shape[2]  # Control variate samples per pixel
    n_std = std_samples.shape[2]  # PT samples per pixel

    for j, i in ti.ndrange(height, width):
        cv_sum = vec3(0.0)
        for k in range(n_cv):
            u = (i + ti.random(ti.f32)) / width
            v = 1 - (j + ti.random(ti.f32)) / height
            ray_origin, ray_direction = camera.generate_ray(u, v)
            ray = Ray(ray_origin, ray_direction)
            F_new = trace_mis(ray, primitives, bvh, lights, light_sampler,
                               sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf,
                               max_depth=scene.max_depth)
            diff = F_new - control[j, i]
            cv_samples[j, i, k] = diff
            cv_sum += diff

        m_cv = cv_sum / n_cv  # Mean difference
        var_cv = vec3(0.0)
        for k in range(n_cv):
            diff = cv_samples[j, i, k] - m_cv
            var_cv += diff * diff
        var_cv /= n_cv

        std_sum = vec3(0.0)
        for k in range(n_std):
            u = (i + ti.random(ti.f32)) / width
            v = 1 - (j + ti.random(ti.f32)) / height
            ray_origin, ray_direction = camera.generate_ray(u, v)
            ray = Ray(ray_origin, ray_direction)
            F_std = trace_mis(ray, primitives, bvh, lights, light_sampler,
                               sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf,
                               max_depth=scene.max_depth)
            std_samples[j, i, k] = F_std
            std_sum += F_std

        m_std = std_sum / n_std
        var_std = vec3(0.0)
        for k in range(n_std):
            diff = std_samples[j, i, k] - m_std
            var_std += diff * diff
        var_std /= n_std

        # Calculate Weights
        epsilon = 1e-6
        w = var_std / (var_cv + var_std + epsilon)  # Per-channel weight (R, G, B)
        w = vec3(ti.min(ti.max(w.x, 0.0), 1.0),
                 ti.min(ti.max(w.y, 0.0), 1.0),
                 ti.min(ti.max(w.z, 0.0), 1.0))

        F_cv = control[j, i] + m_cv  # adjusted control variate estimator
        image[j, i] = w * F_cv + (vec3(1.0) - w) * m_std  # weighted combination



# # Allocate Taichi fields for storing per-pixel samples
# cv_samples = ti.field(dtype=ti.types.vector(3, ti.f32), shape=(camera.height, camera.width, 32))
# std_samples = ti.field(dtype=ti.types.vector(3, ti.f32), shape=(camera.height, camera.width, 32))
