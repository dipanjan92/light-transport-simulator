import taichi as ti
import numpy as np

@ti.dataclass
class Range:
    # A fixed-size vector to hold a slice of random numbers
    values: ti.types.vector(9, ti.f32)  # assuming total_randoms is 9

@ti.dataclass
class Sampler:
    i: ti.i32
    j: ti.i32
    k: ti.i32
    counter: ti.i32
    total: ti.i32
    rng: Range

    @ti.func
    def get_1d(self) -> ti.f32:
        val = 0.0
        if self.counter >= self.total:
            print("Exceeded random sample limit!!!")
        else:
            val = self.rng.values[self.counter]
            self.counter += 1
        return val

@ti.kernel
def initialize_samplers(samplers: ti.template(), total: ti.i32):
    for j, i, k in samplers:
        samplers[j, i, k].i = i
        samplers[j, i, k].j = j
        samplers[j, i, k].k = k
        samplers[j, i, k].counter = 0
        samplers[j, i, k].total = total
        for l in range(total):
            samplers[j, i, k].rng.values[l] = sampler_rand_field[j, i, k, l]

def create_sampler_field(height: int, width: int, spp: int, total_randoms: int, seed: int = 42):
    np.random.seed(seed)
    np_rand_vals = np.random.rand(height, width, spp, total_randoms).astype(np.float32)
    
    global sampler_rand_field
    sampler_rand_field = ti.field(dtype=ti.f32)
    ti.root.dense(ti.j, height).dense(ti.i, width).dense(ti.k, spp).dense(ti.l, total_randoms).place(sampler_rand_field)
    sampler_rand_field.from_numpy(np_rand_vals)
    
    samplers = Sampler.field(shape=(height, width, spp))
    initialize_samplers(samplers, total_randoms)
    return samplers