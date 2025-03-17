import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from samplers.samplers import SobolSampler, SOBOL_OWEN
from samplers.low_discrepancy import SobolMatrix, init_primes, SobolMatrixSize
from samplers.hash import hash_pixel, mix_bits

ti.init(arch=ti.cpu)


sobol_matrix = SobolMatrix()
init_primes(sobol_matrix.primes)
sobol_matrix.init_sobol_vectors()


GRID_W = 100
GRID_H = 100
SAMPLES_PER_PIXEL = 4  # number of samples per pixel (can be looped over later)

scale_field     = ti.field(dtype=ti.i32, shape=(GRID_W, GRID_H))
dimension_field = ti.field(dtype=ti.i32, shape=(GRID_W, GRID_H))
sobol_index_field = ti.field(dtype=ti.u64, shape=(GRID_W, GRID_H))
pixel_field     = ti.Vector.field(2, dtype=ti.i32, shape=(GRID_W, GRID_H))
seed_field      = ti.field(dtype=ti.u64, shape=(GRID_W, GRID_H))

# Output image (we use one float per pixel for demonstration)
image_field     = ti.field(dtype=ti.f32, shape=(GRID_W, GRID_H))


@ti.kernel
def init_pixel_samplers(base_seed: ti.u64):
    for i, j in seed_field:
        pixel_field[i, j] = ti.Vector([i, j])
        seed_field[i, j] = hash_pixel(pixel_field[i, j], base_seed)
        scale_field[i, j] = 4       # as in your sample code
        dimension_field[i, j] = 0
        sobol_index_field[i, j] = 1

BASE_SEED = 559557
init_pixel_samplers(BASE_SEED)


@ti.kernel
def render_kernel():
    for i, j in ti.ndrange(GRID_W, GRID_H):
        # Instantiate the SobolSampler for this pixel.
        sampler = SobolSampler(
            samples_per_pixel=SAMPLES_PER_PIXEL,
            i=i, j=j,
            scale_field=scale_field,
            dimension_field=dimension_field,
            sobol_index_field=sobol_index_field,
            pixel_field=pixel_field,
            seed_field=seed_field,
            sobol_matrix=sobol_matrix,
            randomize_strategy=SOBOL_OWEN,
            seed=BASE_SEED
        )
        # Start the sample for pixel (i, j) with sample index 0 and initial dimension 0.
        sampler.start_pixel_sample(ti.Vector([i, j]), 0, 0)
        # Retrieve one 1D sample and one 2D sample.
        sample1d = sampler.get_1d()
        sample2d = sampler.get_2d()
        # Combine the samples into a pixel intensity.
        image_field[i, j] = (sample1d + (sample2d.x + sample2d.y) * 0.5) * 0.5

# Run the render kernel.
render_kernel()

# Retrieve the output image as a NumPy array and display it.
img_np = image_field.to_numpy()
plt.figure(figsize=(6,6))
plt.imshow(img_np, cmap='viridis', origin='lower')
plt.title("High-Def Sobol Sampled Image")
plt.colorbar()
plt.show()










import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from samplers.samplers import HaltonSampler, HaltonSamplerConfig, SOBOL_OWEN
from samplers.hash import hash_pixel

ti.init(arch=ti.cpu)

# Grid parameters
GRID_W = 100
GRID_H = 100
SAMPLES_PER_PIXEL = 4

# Allocate per-pixel state fields.
dimension_field    = ti.field(dtype=ti.i32, shape=(GRID_W, GRID_H))
halton_index_field = ti.field(dtype=ti.u64, shape=(GRID_W, GRID_H))
pixel_field        = ti.Vector.field(2, dtype=ti.i32, shape=(GRID_W, GRID_H))
seed_field         = ti.field(dtype=ti.u64, shape=(GRID_W, GRID_H))
image_field        = ti.field(dtype=ti.f32, shape=(GRID_W, GRID_H))

@ti.kernel
def init_halton_samplers(base_seed: ti.u64):
    for i, j in dimension_field:
        pixel_field[i, j] = ti.Vector([i, j])
        seed_field[i, j] = hash_pixel(pixel_field[i, j], base_seed)
        dimension_field[i, j] = 0
        halton_index_field[i, j] = 0

BASE_SEED = 559557
init_halton_samplers(BASE_SEED)

# Precompute the configuration on the CPU.
full_res = (GRID_W, GRID_H)
max_halton_resolution = 128
max_prime = 10
halton_config = HaltonSamplerConfig(full_res, max_halton_resolution, max_prime)

@ti.kernel
def render_kernel_halton():
    for i, j in ti.ndrange(GRID_W, GRID_H):
        sampler = HaltonSampler(
            samples_per_pixel=SAMPLES_PER_PIXEL,
            config=halton_config,
            randomize_strategy=SOBOL_OWEN,  # Use one of the defined strategies.
            seed=BASE_SEED,
            dimension_field=dimension_field,
            halton_index_field=halton_index_field,
            pixel_field=pixel_field,
            seed_field=seed_field,
            i=i, j=j
        )
        sampler.start_pixel_sample(ti.Vector([i, j]), 0, 0)
        sample1d = sampler.get_1d()
        sample2d = sampler.get_2d()
        image_field[i, j] = (sample1d + (sample2d.x + sample2d.y) * 0.5) * 0.5

render_kernel_halton()
img_np = image_field.to_numpy()
plt.figure(figsize=(6,6))
plt.imshow(img_np, cmap='viridis', origin='lower')
plt.title("High-Def Halton Sampled Image")
plt.colorbar()
plt.show()