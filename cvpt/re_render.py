from base.bsdf import BSDF
from cvpt.cvpt import trace_cvpt
from integrators.mis_pt import trace_mis
from taichi.math import vec2, vec3, dot, max, exp, sqrt
from base.lights import UniformLightSampler
from primitives.ray import Ray, spawn_ray


"""
This module implements the re-rendering pipeline for dual-material integrators in a light transport simulation.
It includes control image rendering, difference estimation, variance computation, NL-Means filtering, and
cross-weighted composite rendering to achieve unbiased final images.
"""


# ---------------------------------------------------------------------
# Global configuration parameters
WIDTH = camera.width
HEIGHT = camera.height
# Number of samples per pixel for the high-quality control image (m in Eq. (12))
SPP_CONTROL = 1024  
# Total number of samples per pixel for the difference pass (n in Eq. (13))
SPP_DIFF = 64  
# For cross-weighting, split difference pass into two halves:
SPP_DIFF_A = SPP_DIFF // 2
SPP_DIFF_B = SPP_DIFF - SPP_DIFF_A

# NL-Means filtering parameters
NL_WINDOW = 7           # Window size for NL-Means (e.g., 7x7)
NL_H_base = 0.1         # Base smoothing strength
# Clamping threshold for differences D₁ (per channel)
D_CLAMP = 10.0

# ---------------------------------------------------------------------
# Global Taichi fields (declared outside kernels)

# Control image H₀ (from original scene, Eq. (12))
H0 = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

# For difference-pass, we render two sets (A and B) for cross-weighting.
# For pass A:
F1_A     = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
D_A      = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
F1_sum_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
F1_sum_sq_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
H1_sum_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
H1_sum_sq_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
D_sum_A  = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
D_sum_sq_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

# For pass B:
F1_B     = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
D_B      = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
F1_sum_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
F1_sum_sq_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
H1_sum_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
H1_sum_sq_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
D_sum_B  = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
D_sum_sq_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

# Final composite image (after cross-weighting)
F_final = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

# Variance fields for each pass (per channel)
var_F1_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
var_H1_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
var_D_A  = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

var_F1_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
var_H1_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
var_D_B  = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

# Filtered variance fields after NL-Means filtering
# Now we use the edited estimator F₁ as reference.
f_var_F1_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
f_var_H1_A = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
f_var_D_A  = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

f_var_F1_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
f_var_H1_B = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))
f_var_D_B  = ti.Vector.field(3, ti.f32, shape=(HEIGHT, WIDTH))

# ---------------------------------------------------------------------
# Function: composite_estimator
# Computes the per-channel composite estimator using filtered variances,
# full covariance (via Eq. (16) and (15)), and if the denominator is low,
# falls back to sample-rate weighting.
@ti.func
def composite_estimator(F1_val: vec3, D_val: vec3, H0_val: vec3,
                        var_F1: vec3, var_H1: vec3, var_D: vec3,
                        spp_control: ti.f32, spp_diff: ti.f32) -> vec3:
    """
    Computes the per-channel composite estimator using filtered variances and covariance,
    falling back to sample-rate weighting when the variance is too low.

    Args:
        F1_val (vec3): Edited estimator values.
        D_val (vec3): Difference between new and old radiance values.
        H0_val (vec3): Control image radiance.
        var_F1 (vec3): Variance of F1.
        var_H1 (vec3): Variance of H1.
        var_D (vec3): Variance of the difference D.
        spp_control (ti.f32): Number of samples for the control image.
        spp_diff (ti.f32): Number of samples for the difference estimator.

    Returns:
        vec3: The composite estimator for each color channel.
    """
    # Compute covariance between F₁ and H₁ per channel (Eq. (16))
    cov_F1_H1 = (var_F1 + var_H1 - var_D) * 0.5
    # Explicit covariance check: ensure cov² <= var_F1 * var_H1 (per channel)
    for c in ti.static(range(3)):
        if var_F1[c] * var_H1[c] > 0.0 and (cov_F1_H1[c] * cov_F1_H1[c] > var_F1[c] * var_H1[c]):
            if cov_F1_H1[c] >= 0:
                cov_F1_H1[c] = sqrt(var_F1[c] * var_H1[c])
            else:
                cov_F1_H1[c] = -sqrt(var_F1[c] * var_H1[c])
    # Covariance between F_cv and F₁ (Eq. (15)), with F_cv = D + H₀ (H₀ constant)
    cov_Fcv_F1 = var_F1 - cov_F1_H1
    comp = vec3(0.0)
    for c in ti.static(range(3)):
        denom = var_F1[c] + var_D[c] - 2.0 * cov_Fcv_F1[c]
        if denom < 1e-6:
            # Sample-rate fallback: weight by the number of samples used in each estimator.
            F_cv_val = D_val[c] + H0_val[c]
            comp[c] = (spp_diff * F1_val[c] + spp_control * F_cv_val) / (spp_diff + spp_control)
        else:
            w1 = (var_D[c] - cov_Fcv_F1[c]) / denom  # optimal weight (Eq. (11))
            F_cv_val = D_val[c] + H0_val[c]
            comp[c] = w1 * F1_val[c] + (1.0 - w1) * F_cv_val
    return comp

# ---------------------------------------------------------------------
# Kernel: render_control_image
# Renders the high-quality control image H₀ (Eq. (12)) using SPP_CONTROL samples.
@ti.kernel
def render_control_image(scene: ti.template(), image_control: ti.template(),
                         camera: ti.template(), primitives: ti.template(),
                         bvh: ti.template(), lights: ti.template()):
    """
    Renders the high-quality control image H₀ using SPP_CONTROL samples.

    Args:
        scene: The scene configuration containing sampling flags and max depth.
        image_control: Taichi field for storing the rendered control image.
        camera: Camera object to generate rays.
        primitives: Scene primitives for intersection tests.
        bvh: Bounding Volume Hierarchy for acceleration.
        lights: List of light sources.
    """
    light_sampler = UniformLightSampler(lights.shape[0])
    height = camera.height
    width = camera.width
    for j, i in ti.ndrange(height, width):
        L = vec3(0.0)
        for s in range(SPP_CONTROL):
            r_u = ti.random()
            r_v = ti.random()
            u = (i + r_u) / width
            v = 1.0 - (j + r_v) / height
            ray_origin, ray_direction = camera.generate_ray(u, v)
            ray = Ray(ray_origin, ray_direction)
            # Render with the original material via MIS (trace_mis)
            L += trace_mis(ray, primitives, bvh, lights, light_sampler,
                           sample_lights=scene.sample_lights,
                           sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)
        image_control[j, i] = L / SPP_CONTROL  # (Eq. (12))

# ---------------------------------------------------------------------
# Kernel: render_diff
# Renders the edited scene for a given pass (using 'spp' samples) and
# accumulates F₁ (edited estimator) and D = (L_new - L_old) for covariance.
@ti.kernel
def render_diff_std(scene: ti.template(), image_F1: ti.template(), image_D: ti.template(),
                                 F1_sum: ti.template(), F1_sum_sq: ti.template(),
                                 H1_sum: ti.template(), H1_sum_sq: ti.template(),
                                 D_sum: ti.template(), D_sum_sq: ti.template(),
                                 camera: ti.template(), primitives: ti.template(),
                                 bvh: ti.template(), lights: ti.template(),
                                 spp: ti.i32, updated_bsdf: BSDF):
    """
    Renders the edited scene for a given pass using a specified number of samples, and accumulates
    the edited estimator (F₁) and the difference (D = L_new - L_old) for variance computation.

    Args:
        scene: The scene configuration containing sampling flags and max depth.
        image_F1: Taichi field to store the edited estimator image.
        image_D: Taichi field to store the difference image.
        F1_sum: Accumulator field for F₁ sums.
        F1_sum_sq: Accumulator field for F₁ squared sums.
        H1_sum: Accumulator field for old radiance sums.
        H1_sum_sq: Accumulator field for old radiance squared sums.
        D_sum: Accumulator field for difference sums.
        D_sum_sq: Accumulator field for difference squared sums.
        camera: Camera object to generate rays.
        primitives: Scene primitives for intersection tests.
        bvh: Bounding Volume Hierarchy for acceleration.
        lights: List of light sources.
        spp (ti.i32): Number of samples per pixel for this pass.
        updated_bsdf: The updated BSDF for dual-material rendering.
    """
    light_sampler = UniformLightSampler(lights.shape[0])
    height = camera.height
    width = camera.width
    for j, i in ti.ndrange(height, width):
        L_F1 = vec3(0.0)
        L_H1 = vec3(0.0)
        L_D  = vec3(0.0)
        F1_sq = vec3(0.0)
        H1_sq = vec3(0.0)
        D_sq  = vec3(0.0)
        for s in range(spp):
            r_u = ti.random()
            r_v = ti.random()
            u = (i + r_u) / camera.width
            v = 1.0 - (j + r_v) / camera.height
            ray_origin, ray_direction = camera.generate_ray(u, v)
            ray = Ray(ray_origin, ray_direction)
            # Dual-material integrator: returns (L_new, L_old)
            L_new, L_old = trace_cvpt(ray, primitives, bvh, lights, light_sampler, updated_bsdf,
                                       sample_lights=scene.sample_lights,
                                       sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)
            # # F₁ estimator via trace_mis for the edited scene
            # F1_sample = trace_edited_mis(ray, primitives, bvh, lights, light_sampler, updated_bsdf,
            #                       sample_lights=scene.sample_lights,
            #                       sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)
            F1_sample = L_new
            L_F1 += F1_sample
            F1_sq += F1_sample * F1_sample
            
            L_H1 += L_old
            H1_sq += L_old * L_old
            D_sample = L_new - L_old
            # Clamp D_sample to avoid extreme differences (fireflies)
            D_sample = ti.min(ti.max(D_sample, vec3(-D_CLAMP)), vec3(D_CLAMP))
            L_D += D_sample
            D_sq += D_sample * D_sample
        n_pass = ti.cast(spp, ti.f32)
        F1_avg = L_F1 / n_pass
        D_avg  = L_D  / n_pass
        image_F1[j, i] = F1_avg
        image_D[j, i] = D_avg
        F1_sum[j, i] = L_F1
        F1_sum_sq[j, i] = F1_sq
        H1_sum[j, i] = L_H1
        H1_sum_sq[j, i] = H1_sq
        D_sum[j, i] = L_D
        D_sum_sq[j, i] = D_sq

# ---------------------------------------------------------------------
# Kernel: compute_variances
# For a given pass, compute per-channel variance from the accumulators.
@ti.kernel
def compute_variances(F1_sum: ti.template(), F1_sum_sq: ti.template(),
                      H1_sum: ti.template(), H1_sum_sq: ti.template(),
                      D_sum: ti.template(), D_sum_sq: ti.template(),
                      var_F1: ti.template(), var_H1: ti.template(), var_D: ti.template(),
                      spp: ti.i32):
    """
    Computes per-channel variances from the accumulated sums and squared sums for F₁, H₁, and D.

    Args:
        F1_sum: Accumulator field for F₁ sums.
        F1_sum_sq: Accumulator field for F₁ squared sums.
        H1_sum: Accumulator field for H₁ sums.
        H1_sum_sq: Accumulator field for H₁ squared sums.
        D_sum: Accumulator field for difference sums.
        D_sum_sq: Accumulator field for difference squared sums.
        var_F1: Output variance field for F₁.
        var_H1: Output variance field for H₁.
        var_D: Output variance field for D.
        spp (ti.i32): Number of samples used in the accumulation.
    """
    height = F1_sum.shape[0]
    width = F1_sum.shape[1]
    n = ti.cast(spp, ti.f32)
    for j, i in ti.ndrange(height, width):
        for c in ti.static(range(3)):
            mean_F1 = F1_sum[j, i][c] / n
            mean_F1_sq = F1_sum_sq[j, i][c] / n
            var_F1[j, i][c] = mean_F1_sq - mean_F1 * mean_F1

            mean_H1 = H1_sum[j, i][c] / n
            mean_H1_sq = H1_sum_sq[j, i][c] / n
            var_H1[j, i][c] = mean_H1_sq - mean_H1 * mean_H1

            mean_D = D_sum[j, i][c] / n
            mean_D_sq = D_sum_sq[j, i][c] / n
            var_D[j, i][c] = mean_D_sq - mean_D * mean_D

# ---------------------------------------------------------------------
# Kernel: nl_means_filter_variance
# NL-Means filter on a vector field (variance per channel) using a parameterized window.
# The filtering uses the noisy edited estimator F₁ (passed as ref_img) for guidance.
# It also adapts the effective NL parameter based on local noise.
@ti.kernel
def nl_means_filter_variance(var_in: ti.template(), var_out: ti.template(), ref_img: ti.template()):
    """
    Applies Non-Local Means filtering to a variance field using a reference image for guidance.
    The filter adapts the effective smoothing strength based on local noise estimation.

    Args:
        var_in: Input variance field (per channel).
        var_out: Output filtered variance field (per channel).
        ref_img: Reference image (edited estimator) used to guide filtering.
    """
    height = var_in.shape[0]
    width = var_in.shape[1]
    
    half = NL_WINDOW // 2  # Half window size for easier indexing

    # Iterate over each pixel in parallel
    for j, i in ti.ndrange(height, width):
        ref_val = ref_img[j, i]  # Get the reference pixel value

        # Step 1: Estimate local noise variance
        local_sum = ti.Vector([0.0, 0.0, 0.0])
        count = 0.0

        for dj, di in ti.ndrange((-half, half + 1), (-half, half + 1)):
            jj = ti.min(ti.max(j + dj, 0), height - 1)
            ii = ti.min(ti.max(i + di, 0), width - 1)
            diff = ref_val - ref_img[jj, ii]
            local_sum += diff * diff  # Sum of squared differences
            count += 1.0

        local_std = ti.sqrt((local_sum.x + local_sum.y + local_sum.z) / (3.0 * count + 1e-8))

        # Step 2: Compute adaptive filtering strength
        effective_NL = NL_H_base * (1.0 + local_std)

        # Step 3: Apply Non-Local Means filtering
        filtered = ti.Vector([0.0, 0.0, 0.0])
        weight_sum = ti.Vector([0.0, 0.0, 0.0])

        for dj, di in ti.ndrange((-half, half + 1), (-half, half + 1)):
            jj = ti.min(ti.max(j + dj, 0), height - 1)
            ii = ti.min(ti.max(i + di, 0), width - 1)
            neigh_val = ref_img[jj, ii]

            diff = ref_val - neigh_val
            dist2 = diff.dot(diff)  # Faster squared Euclidean distance
            w = ti.exp(-dist2 / (effective_NL * effective_NL + 1e-8))  # Exponential weight

            filtered += var_in[jj, ii] * w
            weight_sum += ti.Vector([w, w, w])  # Sum weights per RGB channel

        # Normalize to avoid intensity shifts
        var_out[j, i] = filtered / (weight_sum + 1e-8)

# ---------------------------------------------------------------------
# Kernel: composite_render_cross
# Using cross-weighting: use filtered variances from pass A to weight pass B and vice versa,
# then average the two composite estimates.
@ti.kernel
def composite_render_cross(scene: ti.template(), image_control: ti.template(),
                           F1_A_img: ti.template(), D_A_img: ti.template(),
                           F1_B_img: ti.template(), D_B_img: ti.template(),
                           var_F1_A_f: ti.template(), var_H1_A_f: ti.template(), var_D_A_f: ti.template(),
                           var_F1_B_f: ti.template(), var_H1_B_f: ti.template(), var_D_B_f: ti.template(),
                           final_img: ti.template()):
    """
    Composites the final image by cross-weighting two passes. Each pass is weighted using the
    filtered variances from the other pass to produce an unbiased composite estimator.

    Args:
        scene: The scene configuration containing sampling flags and max depth.
        image_control: Control image H₀.
        F1_A_img: Edited estimator image from pass A.
        D_A_img: Difference image from pass A.
        F1_B_img: Edited estimator image from pass B.
        D_B_img: Difference image from pass B.
        var_F1_A_f: Filtered variance field for F₁ from pass A.
        var_H1_A_f: Filtered variance field for H₁ from pass A.
        var_D_A_f: Filtered variance field for D from pass A.
        var_F1_B_f: Filtered variance field for F₁ from pass B.
        var_H1_B_f: Filtered variance field for H₁ from pass B.
        var_D_B_f: Filtered variance field for D from pass B.
        final_img: Output field for the final composite image.
    """
    height = final_img.shape[0]
    width = final_img.shape[1]
    spp_control_f = ti.cast(SPP_CONTROL, ti.f32)
    spp_diff_f_A = ti.cast(SPP_DIFF_A, ti.f32)
    spp_diff_f_B = ti.cast(SPP_DIFF_B, ti.f32)
    for j, i in ti.ndrange(height, width):
        # Composite using pass B estimates weighted by pass A filtered variances:
        comp_B = composite_estimator(F1_B_img[j, i], D_B_img[j, i], image_control[j, i],
                                     var_F1_A_f[j, i], var_H1_A_f[j, i], var_D_A_f[j, i],
                                     spp_control_f, spp_diff_f_B)
        # Composite using pass A estimates weighted by pass B filtered variances:
        comp_A = composite_estimator(F1_A_img[j, i], D_A_img[j, i], image_control[j, i],
                                     var_F1_B_f[j, i], var_H1_B_f[j, i], var_D_B_f[j, i],
                                     spp_control_f, spp_diff_f_A)
        final_img[j, i] = (comp_A + comp_B) * 0.5

# ---------------------------------------------------------------------
# Main pipeline function: run_rerendering_pipeline
# This function runs:
# 1. Control image rendering.
# 2. Two passes for difference and standard estimators (using the generic kernel).
# 3. Variance computation for each pass.
# 4. NL-Means filtering on variance fields using F₁ as reference.
# 5. Cross-weighted composite rendering.
def run_rerendering_pipeline(scene, camera, primitives, bvh, lights, bsdf_new):
    """
    Executes the complete re-rendering pipeline which includes:
    1. Rendering the high-quality unedited control image (H₀).
    2. Rendering two passes for difference estimation using a dual-material integrator.
    3. Computing variances for each pass.
    4. Applying NL-Means filtering to the variance fields.
    5. Compositing the two passes using cross-weighting to obtain the final image.

    Args:
        scene: The scene configuration containing sampling flags and max depth.
        camera: Camera object to generate rays.
        primitives: Scene primitives for intersection tests.
        bvh: Bounding Volume Hierarchy for acceleration.
        lights: List of light sources.
        bsdf_new: The updated BSDF for dual-material rendering.
    """
    start_time = time.time()
    # Render high-quality control image H0
    print("Rendering Unedited Control Image...")
    render_control_image(scene, H0, camera, primitives, bvh, lights)
    ti.sync()
    time_control_image = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_control_image))}")

    start_time = time.time()
    # Render difference and standard estimators for pass A using spp_diff_A samples
    print("Rendering Pass A...")
    render_diff_std(scene, F1_A, D_A, F1_sum_A, F1_sum_sq_A,
                                 H1_sum_A, H1_sum_sq_A, D_sum_A, D_sum_sq_A,
                                 camera, primitives, bvh, lights, SPP_DIFF_A, bsdf_new)
    ti.sync()
    time_pass_a = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_pass_a))}")

    start_time = time.time()
    # Render difference and standard estimators for pass B using spp_diff_B samples
    print("Rendering Pass B...")
    render_diff_std(scene, F1_B, D_B, F1_sum_B, F1_sum_sq_B,
                                 H1_sum_B, H1_sum_sq_B, D_sum_B, D_sum_sq_B,
                                 camera, primitives, bvh, lights, SPP_DIFF_B, bsdf_new)
    ti.sync()
    time_pass_b = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_pass_b))}")
    
    start_time = time.time()
    # Compute per-channel variances for both passes
    print("Computing Variances for Pass A...")
    compute_variances(F1_sum_A, F1_sum_sq_A, H1_sum_A, H1_sum_sq_A, D_sum_A, D_sum_sq_A,
                      var_F1_A, var_H1_A, var_D_A, SPP_DIFF_A)
    ti.sync()
    time_variance_a = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_variance_a))}")
    
    start_time = time.time()
    print("Computing Variances for Pass B...")
    compute_variances(F1_sum_B, F1_sum_sq_B, H1_sum_B, H1_sum_sq_B, D_sum_B, D_sum_sq_B,
                      var_F1_B, var_H1_B, var_D_B, SPP_DIFF_B)
    ti.sync()
    time_variance_b = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_variance_b))}")
    
    start_time = time.time()
    # Apply NL-Means filtering on the variance fields using the edited estimator F1 as reference
    print("Filtering Variances...")
    nl_means_filter_variance(var_F1_A, f_var_F1_A, F1_A)
    nl_means_filter_variance(var_H1_A, f_var_H1_A, F1_A)
    nl_means_filter_variance(var_D_A, f_var_D_A, F1_A)
    nl_means_filter_variance(var_F1_B, f_var_F1_B, F1_B)
    nl_means_filter_variance(var_H1_B, f_var_H1_B, F1_B)
    nl_means_filter_variance(var_D_B, f_var_D_B, F1_B)
    ti.sync()
    time_filtering = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_filtering))}")
    
    start_time = time.time()
    # Finally, composite the two passes using cross-weighting for unbiased results
    print("Cross-Weighting Results...")
    composite_render_cross(scene, H0, F1_A, D_A, F1_B, D_B,
                             f_var_F1_A, f_var_H1_A, f_var_D_A,
                             f_var_F1_B, f_var_H1_B, f_var_D_B,
                             F_final)
    ti.sync()
    time_cross_weighting = time.time() - start_time
    print(f"  Time taken: {time.strftime('%M:%S', time.gmtime(time_cross_weighting))}")


# ---------------------------------------------------------------------
# End of complete re-rendering pipeline code.