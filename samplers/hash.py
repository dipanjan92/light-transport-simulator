import taichi as ti

@ti.func
def hash_combine(seed: ti.u64, v: ti.u64):
    seed ^= v + ti.u64(0x9e3779b97f4a7c15) + (seed << 6) + (seed >> 2)
    return seed

@ti.func
def hash_pixel(p, seed):
    h = ti.u64(seed)
    h = hash_combine(h, ti.u64(p[0]))
    h = hash_combine(h, ti.u64(p[1]))
    return h

@ti.func
def mix_bits(v: ti.u64) -> ti.u64:
    # Ensure v is treated as a 64-bit unsigned integer
    v = ti.u64(v)
    v ^= (v >> 21)
    v ^= v << 37
    v ^= (v >> 4)
    v *= ti.u64(2685821657736338717)
    v ^= v >> 32
    return v

@ti.func
def pbrt_permute(x: ti.u32, l: ti.u32, seed: ti.u32) -> ti.u32:
    w = (l - 1) | 1
    x = ((x ^ seed) * 0x7fffffff) & 0xffffffff
    return ((x >> 16) | (x << 16)) & w
