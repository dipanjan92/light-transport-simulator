from integrators.mis_pt import trace_mis
from primitives.ray import Ray
from base.lights import UniformLightSampler
import taichi as ti
from taichi.math import vec3

# -------------------------------------------------------
# Data structures & Fields
# -------------------------------------------------------
@ti.dataclass
class CVStats:
    F_sum: ti.types.vector(3, ti.f32)
    H_sum: ti.types.vector(3, ti.f32)
    D_sum: ti.types.vector(3, ti.f32)
    FF_sum: ti.types.vector(3, ti.f32)
    HH_sum: ti.types.vector(3, ti.f32)
    FH_sum: ti.types.vector(3, ti.f32)
    count: ti.i32

# Example resolution / sample settings (adjust to your scene):
scene_spp = 64  # total SPP (split half for pass1, half for pass2)
res_y = 400
res_x = 400

# Global fields
stats1 = CVStats.field(shape=(res_y, res_x))   # pass1 stats
stats2 = CVStats.field(shape=(res_y, res_x))   # pass2 stats

# Pass‐1 color buffer (guides NL‐Means):
pass1_color = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))

# Weighted combination factors (the final w per channel):
weights = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))

# Final image (after pass2 + compositing):
image = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))

# Temporary unfiltered var/cov fields:
varF_unf = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))
varH_unf = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))
covFH_unf = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))

# Filtered var/cov fields after patch‐based NL‐Means:
varF_flt = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))
varH_flt = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))
covFH_flt = ti.Vector.field(3, ti.f32, shape=(res_y, res_x))


# -------------------------------------------------------
# 0) Clear stats
# -------------------------------------------------------
@ti.kernel
def clear_stats(stats: ti.template()):
    stats.F_sum.fill(0.0)
    stats.H_sum.fill(0.0)
    stats.D_sum.fill(0.0)
    stats.FF_sum.fill(0.0)
    stats.HH_sum.fill(0.0)
    stats.FH_sum.fill(0.0)
    stats.count.fill(0)


# -------------------------------------------------------
# 1) Render half the samples for pass1 or pass2
#     Now using two primitives lists:
#         primitives_old for the "old" path (control material)
#         primitives_new for the "new" path (edited material)
# -------------------------------------------------------
@ti.kernel
def render_control_variate_half(
    scene: ti.template(),
    out_image: ti.template(),   # pass1_color or pass2 image
    control_img: ti.template(),
    stats: ti.template(),
    lights: ti.template(),
    camera: ti.template(),
    primitives_old: ti.template(),
    primitives_new: ti.template(),
    bvh: ti.template(),
    start_spp: ti.i32,
    end_spp: ti.i32
):
    light_sampler = UniformLightSampler(lights.shape[0])
    height, width = camera.height, camera.width

    ti.loop_config(parallelize=4, block_dim=16)
    for j, i in ti.ndrange(height, width):
        F_accum = ti.Vector([0.0, 0.0, 0.0])
        H_accum = ti.Vector([0.0, 0.0, 0.0])
        D_accum = ti.Vector([0.0, 0.0, 0.0])
        FF_accum = ti.Vector([0.0, 0.0, 0.0])
        HH_accum = ti.Vector([0.0, 0.0, 0.0])
        FH_accum = ti.Vector([0.0, 0.0, 0.0])

        # (Optional) Read control image if needed.
        # c = control_img[j, i]

        for s in range(start_spp, end_spp):
            u = (i + ti.random()) / width
            v = 1.0 - (j + ti.random()) / height

            # "Old" path (control material):
            ray_org2, ray_dir2 = camera.generate_ray(u, v)
            ray_old = Ray(ray_org2, ray_dir2)
            H = trace_mis(ray_old, primitives_old, bvh, lights, light_sampler,
                          scene.sample_lights, scene.sample_bsdf, scene.max_depth)

            # "New" path (edited material):
            ray_org, ray_dir = camera.generate_ray(u, v)
            ray_new = Ray(ray_org, ray_dir)
            F = trace_mis(ray_new, primitives_new, bvh, lights, light_sampler,
                          scene.sample_lights, scene.sample_bsdf, scene.max_depth)

            D = F - H
            F_accum += F
            H_accum += H
            D_accum += D
            FF_accum += F * F
            HH_accum += H * H
            FH_accum += F * H

        nsamples = (end_spp - start_spp)
        stats[j, i].F_sum += F_accum
        stats[j, i].H_sum += H_accum
        stats[j, i].D_sum += D_accum
        stats[j, i].FF_sum += FF_accum
        stats[j, i].HH_sum += HH_accum
        stats[j, i].FH_sum += FH_accum
        stats[j, i].count += nsamples

        out_image[j, i] += (F_accum / float(nsamples))
    # return  # single return


# -------------------------------------------------------
# 2) Compute unfiltered Var(F), Var(H), Cov(F,H)
# -------------------------------------------------------
@ti.kernel
def compute_var_covar(
    stats: ti.template(),
    varF_out: ti.template(),
    varH_out: ti.template(),
    covFH_out: ti.template()
):
    for j, i in stats:
        s = stats[j, i]
        n = s.count
        if n > 1:
            varF = (s.FF_sum - s.F_sum * (s.F_sum / n)) / float(n - 1)
            varH = (s.HH_sum - s.H_sum * (s.H_sum / n)) / float(n - 1)
            covFH = (s.FH_sum - s.F_sum * (s.H_sum / n)) / float(n - 1)
            varF_out[j, i] = varF
            varH_out[j, i] = varH
            covFH_out[j, i] = covFH
        else:
            varF_out[j, i] = ti.Vector([0.0, 0.0, 0.0])
            varH_out[j, i] = ti.Vector([0.0, 0.0, 0.0])
            covFH_out[j, i] = ti.Vector([0.0, 0.0, 0.0])
    # return


# -------------------------------------------------------
# 3) Patch-based NL‐Means for var/covar (exact from paper)
# -------------------------------------------------------
@ti.kernel
def nlmeans_filter_var_covar_patch(
    pass1_col: ti.template(),
    varF_in: ti.template(),
    varH_in: ti.template(),
    covFH_in: ti.template(),
    varF_out: ti.template(),
    varH_out: ti.template(),
    covFH_out: ti.template(),
    search_window: ti.i32,     # e.g., 10 => +/- 10
    patch_radius: ti.i32,      # e.g., 3 => 7x7 patch
    h: ti.f32
):
    height, width = pass1_col.shape

    ti.loop_config(parallelize=4, block_dim=16)
    for j, i in ti.ndrange(height, width):
        wsum = 0.0
        varF_accum = ti.Vector([0.0, 0.0, 0.0])
        varH_accum = ti.Vector([0.0, 0.0, 0.0])
        covFH_accum = ti.Vector([0.0, 0.0, 0.0])
        # Loop over neighbor pixels in the search region
        for dy in range(-search_window, search_window + 1):
            for dx in range(-search_window, search_window + 1):
                ny = j + dy
                nx = i + dx
                if 0 <= ny < height and 0 <= nx < width:
                    patchDist2 = 0.0
                    # For each offset in the patch:
                    for py in range(-patch_radius, patch_radius + 1):
                        for px in range(-patch_radius, patch_radius + 1):
                            cy = j + py
                            cx = i + px
                            ny2 = ny + py
                            nx2 = nx + px
                            if (0 <= cy < height and 0 <= cx < width and
                                0 <= ny2 < height and 0 <= nx2 < width):
                                colCenter = pass1_col[cy, cx]
                                colNeighbor = pass1_col[ny2, nx2]
                                diff = colCenter - colNeighbor
                                patchDist2 += diff.dot(diff)
                    weight = ti.exp(-patchDist2 / (h * h))
                    wsum += weight
                    varF_accum += varF_in[ny, nx] * weight
                    varH_accum += varH_in[ny, nx] * weight
                    covFH_accum += covFH_in[ny, nx] * weight
        inv_wsum = 1.0 / (wsum + 1e-12)
        varF_out[j, i] = varF_accum * inv_wsum
        varH_out[j, i] = varH_accum * inv_wsum
        covFH_out[j, i] = covFH_accum * inv_wsum
    # return  # single return


# -------------------------------------------------------
# 4) Compute final per‐channel weights from filtered var/cov using full covariance
#    Optimal weight: w = (Var(H) - Cov(F,H)) / (Var(F) + Var(H) - 2*Cov(F,H))
# -------------------------------------------------------
@ti.kernel
def compute_optimal_weights_full(
    varF: ti.template(),
    varH: ti.template(),
    covFH: ti.template(),
    weights_out: ti.template()
):
    height, width = varF.shape

    ti.loop_config(parallelize=4, block_dim=16)
    for j, i in ti.ndrange(height, width):
        w = ti.Vector([0.0, 0.0, 0.0])
        vf = varF[j, i]
        vh = varH[j, i]
        cfh = covFH[j, i]
        # For each color channel, compute the weight using the full covariance formula.
        # Denom = Var(F) + Var(H) - 2 * Cov(F, H)
        for c in ti.static(range(3)):
            denom = vf[c] + vh[c] - 2.0 * cfh[c] + 1e-8
            if denom > 1e-8:
                w[c] = (vh[c] - cfh[c]) / denom
            else:
                w[c] = 0.5
        weights_out[j, i] = w
    # return


# -------------------------------------------------------
# 5a) Plain CVPT: composite without weighting (raw control variate)
# -------------------------------------------------------
@ti.kernel
def apply_cvpt(
    stats2: ti.template(),
    control_img: ti.template(),
    out_image: ti.template()
):
    height, width = out_image.shape

    ti.loop_config(parallelize=4, block_dim=16)
    for j, i in ti.ndrange(height, width):
        c2 = stats2[j, i].count
        if c2 == 0:
            out_image[j, i] = control_img[j, i]
        else:
            D_mean = stats2[j, i].D_sum / float(c2)
            out_image[j, i] = control_img[j, i] + D_mean
    # return


# -------------------------------------------------------
# 5b) CVPT-opt: optimal composite using computed weights
# -------------------------------------------------------
@ti.kernel
def apply_composite_estimator_cross(
    stats2: ti.template(),
    weights_in: ti.template(),
    control_img: ti.template(),
    out_image: ti.template()
):
    height, width = out_image.shape

    ti.loop_config(parallelize=4, block_dim=16)
    for j, i in ti.ndrange(height, width):
        c2 = stats2[j, i].count
        if c2 == 0:
            out_image[j, i] = control_img[j, i]
        else:
            F_mean = stats2[j, i].F_sum / float(c2)
            D_mean = stats2[j, i].D_sum / float(c2)
            # CV estimate: control + D
            F_cv = control_img[j, i] + D_mean
            w = weights_in[j, i]
            out_image[j, i] = w * F_cv + (ti.Vector([1.0, 1.0, 1.0]) - w) * F_mean
    # return


# -------------------------------------------------------
# 6) Putting it all together
# -------------------------------------------------------
def main(scene, lights, camera, primitives_old, primitives_new, bvh, control, use_optimal=True):
    # 0) Clear:
    clear_stats(stats1)
    clear_stats(stats2)
    pass1_color.fill(ti.Vector([0.0, 0.0, 0.0]))
    image.fill(ti.Vector([0.0, 0.0, 0.0]))

    half_spp = scene_spp // 2

    # 1) Pass1: gather stats. Store naive F in pass1_color.
    # Use primitives_old for the control path and primitives_new for the new path.
    render_control_variate_half(
        scene, pass1_color, control, stats1,
        lights, camera, primitives_old, primitives_new, bvh,
        0, half_spp
    )

    # 2) Compute unfiltered var/cov from pass1 stats.
    compute_var_covar(stats1, varF_unf, varH_unf, covFH_unf)

    # 3) Patch‐based NL‐Means filter.
    search_window = 10
    patch_radius = 3
    h_param = 0.1

    nlmeans_filter_var_covar_patch(
        pass1_color,
        varF_unf, varH_unf, covFH_unf,
        varF_flt, varH_flt, covFH_flt,
        search_window, patch_radius, h_param
    )

    # 4) Compute final weights from filtered var/cov using full covariance.
    compute_optimal_weights_full(varF_flt, varH_flt, covFH_flt, weights)

    # 5) Pass2: gather stats from second half. Write naive F in 'image'.
    render_control_variate_half(
        scene, image, control, stats2,
        lights, camera, primitives_new, primitives_new, bvh,
        half_spp, scene_spp
    )

    # 6) Final composite:
    if use_optimal:
        # Use CVPT-opt:
        apply_composite_estimator_cross(stats2, weights, control, image)
    else:
        # Use plain CVPT (raw control variate)
        apply_cvpt(stats2, control, image)

    # 'image' now contains the final result.
    # return  # single return
