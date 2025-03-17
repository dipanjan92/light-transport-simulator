import taichi as ti
from samplers.hash import mix_bits, hash_pixel
from samplers.low_discrepancy import (
    radical_inverse, 
    scrambled_radical_inverse, 
    inverse_radical_inverse, 
    owen_scrambled_radical_inverse, 
    init_primes, 
    SobolMatrix, 
    SobolMatrixSize,
    sobol_fast_owen_bits,
    sobol_full_owen_bits,
    sobol_permute_bits
)

# Constants for randomization strategies
SOBOL_RANDOMIZE_NONE = 0
SOBOL_PERMUTE_DIGITS = 1
SOBOL_FAST_OWEN = 2
SOBOL_OWEN = 3

class HaltonSamplerConfig:
    def __init__(self, full_res, max_halton_resolution, max_prime):
        # full_res should be a tuple, e.g. (width, height)
        self.full_res = full_res
        self.max_halton_resolution = max_halton_resolution
        self.max_prime = max_prime

        # Compute base scales and exponents
        self.base_scales = [0, 0]
        self.base_exponents = [0, 0]
        for d in range(2):
            base = 2 if d == 0 else 3
            scale = 1
            exp = 0
            while scale < min(full_res[d], max_halton_resolution):
                scale *= base
                exp += 1
            self.base_scales[d] = scale
            self.base_exponents[d] = exp

        # Compute multiplicative inverses.
        def multiplicative_inverse(a, n):
            for x in range(n):
                if (a * x) % n == 1:
                    return x
            return 1
        self.mult_inverse = [
            multiplicative_inverse(self.base_scales[1], self.base_scales[0]),
            multiplicative_inverse(self.base_scales[0], self.base_scales[1])
        ]

        # Instead of hardcoding, create a Taichi field for primes and initialize it using init_primes.
        self.primes = ti.field(ti.i32, shape=(max_prime,))
        init_primes(self.primes)

# Global constants (adjust as needed)
MAX_HALTON_RESOLUTION = 128
PrimeTableSize = 1000  # Must be >= required number of primes

@ti.data_oriented
class HaltonSampler:
    def __init__(self, samples_per_pixel: int, config: HaltonSamplerConfig,
                 randomize_strategy: int, seed: int,
                 dimension_field, halton_index_field, pixel_field, seed_field,
                 i: ti.i32, j: ti.i32):
        
        self.samples_per_pixel = samples_per_pixel
        self.config = config
        self.randomize = randomize_strategy
        self.seed = seed
        self.i = i
        self.j = j

        # Store references to the per-pixel state fields.
        self.dimension_field = dimension_field
        self.halton_index_field = halton_index_field
        self.pixel_field = pixel_field
        self.seed_field = seed_field

    @ti.func
    def _multiplicative_inverse(self, a: ti.i32, n: ti.i32) -> ti.i32:
        # Extended Euclidean algorithm (single return at end)
        t = ti.i32(0)
        newt = ti.i32(1)
        r = n
        newr = a
        while newr != 0:
            quotient = r / newr
            temp = newt
            newt = t - quotient * newt
            t = temp
            temp = newr
            newr = r - quotient * newr
            r = temp
        result = t
        if result < 0:
            result += n
        return result

    @ti.func
    def start_pixel_sample(self, p: ti.types.vector(2, ti.i32), sample_index: ti.i32, dim: ti.i32):
        # Store pixel coordinate in the appropriate slice.
        self.pixel_field[self.i, self.j] = p
        # Compute sample stride.
        sample_stride = self.config.base_scales[0] * self.config.base_scales[1]
        halton_index = ti.u64(0)
        if sample_stride > 1:
            # take pixel coordinate modulo MAX_HALTON_RESOLUTION.
            pm0 = p.x % MAX_HALTON_RESOLUTION
            pm1 = p.y % MAX_HALTON_RESOLUTION
            # For dimension 0 (base 2)
            offset0 = ti.u64(pm0)
            n0 = inverse_radical_inverse(2, self.config.base_exponents[0], offset0)
            halton_index += n0 * (sample_stride // self.config.base_scales[0]) * ti.u64(self.config.mult_inverse[0])
            # For dimension 1 (base 3)
            offset1 = ti.u64(pm1)
            n1 = inverse_radical_inverse(3, self.config.base_exponents[1], offset1)
            halton_index += n1 * (sample_stride // self.config.base_scales[1]) * ti.u64(self.config.mult_inverse[1])
            halton_index = halton_index % ti.u64(sample_stride)
        halton_index += ti.u64(sample_index) * ti.u64(sample_stride)
        self.halton_index_field[self.i, self.j] = halton_index
        self.dimension_field[self.i, self.j] = ti.max(2, dim)

    @ti.func
    def get_1d(self) -> ti.f32:
        d = self.dimension_field[self.i, self.j]
        self.dimension_field[self.i, self.j] = d + 1
        result = ti.f32(0.0)
        if self.randomize == 0:
            result = radical_inverse(self.config.primes[d],
                                     self.halton_index_field[self.i, self.j])
        elif self.randomize == 1:
            result = scrambled_radical_inverse(self.config.primes[d],
                                               self.halton_index_field[self.i, self.j],
                                               ti.u32(mix_bits(ti.u64(self.seed))))
        else:
            result = owen_scrambled_radical_inverse(self.config.primes[d],
                                                     self.halton_index_field[self.i, self.j],
                                                     ti.u32(mix_bits(ti.u64(self.seed))))
        return result

    @ti.func
    def get_2d(self) -> ti.types.vector(2, ti.f32):
        d = self.dimension_field[self.i, self.j]
        self.dimension_field[self.i, self.j] = d + 2
        ret = ti.Vector([ti.f32(0.0), ti.f32(0.0)])
        if self.randomize == 0:
            ret[0] = radical_inverse(self.config.primes[d],
                                     self.halton_index_field[self.i, self.j])
            ret[1] = radical_inverse(self.config.primes[d+1],
                                     self.halton_index_field[self.i, self.j])
        elif self.randomize == 1:
            ret[0] = scrambled_radical_inverse(self.config.primes[d],
                                               self.halton_index_field[self.i, self.j],
                                               ti.u32(mix_bits(ti.u64(self.seed))))
            ret[1] = scrambled_radical_inverse(self.config.primes[d+1],
                                               self.halton_index_field[self.i, self.j],
                                               ti.u32(mix_bits(ti.u64(self.seed))))
        else:
            ret[0] = owen_scrambled_radical_inverse(self.config.primes[d],
                                                    self.halton_index_field[self.i, self.j],
                                                    ti.u32(mix_bits(ti.u64(self.seed))))
            ret[1] = owen_scrambled_radical_inverse(self.config.primes[d+1],
                                                    self.halton_index_field[self.i, self.j],
                                                    ti.u32(mix_bits(ti.u64(self.seed))))
        return ret





@ti.data_oriented
class SobolSampler:
    def __init__(self, samples_per_pixel: int,
                 i: ti.i32, j: ti.i32,
                 scale_field, dimension_field, sobol_index_field, pixel_field, seed_field,
                 sobol_matrix: SobolMatrix,
                 randomize_strategy: int,  
                 seed: int                 
    ):
        self.samples_per_pixel = samples_per_pixel
        self.i = i
        self.j = j
        self.scale_field = scale_field
        self.dimension_field = dimension_field
        self.sobol_index_field = sobol_index_field
        self.pixel_field = pixel_field
        self.seed_field = seed_field
        self.sobol_matrix = sobol_matrix

        self.randomize_strategy = randomize_strategy
        self.seed = seed

    @ti.func
    def start_pixel_sample(self, p: ti.types.vector(2, ti.i32), sample_index: ti.i32, dim: ti.i32):
        self.pixel_field[self.i, self.j] = p
        log2_scale = 0
        scale_temp = self.scale_field[self.i, self.j]
        while scale_temp > 1:
            scale_temp >>= 1
            log2_scale += 1

        # Use the existing logic to compute sobol_index
        index = self.sobol_interval_to_index(log2_scale, ti.u64(sample_index), p)
        # Incorporate the pixel seed
        index ^= mix_bits(self.seed_field[self.i, self.j])
        index += ti.u64(sample_index)
        self.sobol_index_field[self.i, self.j] = index

        # dimension = 2 + sample_index * 3
        self.dimension_field[self.i, self.j] = ti.max(2, dim) + sample_index * 3

    @ti.func
    def sobol_interval_to_index(self, log2_scale: ti.i32, sample_index: ti.u64,
                                p: ti.types.vector(2, ti.i32)) -> ti.u64:
        # Unchanged from your existing code
        index = sample_index
        if log2_scale > 0:
            m2 = log2_scale << 1
            index = sample_index << m2
            delta = ti.u64(0)
            frame = sample_index
            for c in range(log2_scale):
                if frame & 1:
                    delta ^= self.sobol_matrix.vdC_sobol_matrices[log2_scale - 1, c]
                frame >>= 1
            b = ((ti.u64(p[0]) << log2_scale) | ti.u64(p[1])) ^ delta
            for c in range(m2):
                if b & 1:
                    index ^= self.sobol_matrix.vdC_sobol_matrices_inv[log2_scale - 1, c]
                b >>= 1
        return index

    @ti.func
    def get_1d(self) -> ti.f32:
        # Check dimension
        if self.dimension_field[self.i, self.j] >= SobolMatrixSize:
            self.dimension_field[self.i, self.j] = 2
        d = self.dimension_field[self.i, self.j]
        self.dimension_field[self.i, self.j] = d + 1
        # Retrieve the sobol_index
        a = self.sobol_index_field[self.i, self.j]
        # Now apply the randomization strategy
        return self.sobol_sample(a, d)

    @ti.func
    def get_2d(self) -> ti.types.vector(2, ti.f32):
        if self.dimension_field[self.i, self.j] + 1 >= SobolMatrixSize:
            self.dimension_field[self.i, self.j] = 2
        d = self.dimension_field[self.i, self.j]
        self.dimension_field[self.i, self.j] = d + 2

        a = self.sobol_index_field[self.i, self.j]
        ret = ti.Vector([self.sobol_sample(a, d),
                         self.sobol_sample(a, d + 1)])
        return ret

    @ti.func
    def sobol_sample(self, a: ti.u64, dimension: ti.i32) -> ti.f32:
        # Single return approach:
        result = ti.f32(0.0)
        scramble_seed = mix_bits(ti.u64(self.seed)) ^ ti.u64(dimension)

        if self.randomize_strategy == SOBOL_RANDOMIZE_NONE:
            # No scramble: just expand bits
            result = self.sobol_sample_raw(a, dimension)
        elif self.randomize_strategy == SOBOL_PERMUTE_DIGITS:
            new_a = sobol_permute_bits(a, scramble_seed)
            result = self.sobol_sample_raw(new_a, dimension)
        elif self.randomize_strategy == SOBOL_FAST_OWEN:
            new_a = sobol_fast_owen_bits(a, scramble_seed)
            result = self.sobol_sample_raw(new_a, dimension)
        else:
            # SOBOL_OWEN
            new_a = sobol_full_owen_bits(a, scramble_seed)
            result = self.sobol_sample_raw(new_a, dimension)

        return result

    @ti.func
    def sobol_sample_raw(self, a: ti.u64, dimension: ti.i32) -> ti.f32:
        """
        Your existing logic that expands 'a' into bits using the sobol_direction_vectors.
        """
        a = a % (ti.u64(1) << SobolMatrixSize)
        v = ti.u32(0)
        i = 0
        while a != 0:
            if a & 1:
                v ^= ti.cast(self.sobol_matrix.sobol_direction_vectors[dimension, i], ti.u32)
            a >>= 1
            i += 1
        return ti.cast(v, ti.f32) * (1.0 / 4294967296.0)