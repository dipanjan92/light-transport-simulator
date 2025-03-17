import taichi as ti

@ti.data_oriented
class Sampler:
    def __init__(self, seed):
        # Use a single field to store RNG state.
        self.state_field = ti.field(dtype=ti.i64, shape=())
        self.state_field[None] = seed

    @ti.func
    def initialize(self):
        self.state_field[None] = ti.i64(self.scramble_seed(self.state_field[None]))

    @ti.func
    def scramble_seed(self, seed):
        seed = (seed ^ 61) ^ (seed >> 16)
        seed *= 9
        seed = seed ^ (seed >> 4)
        seed *= 0x27d4eb2d
        seed = seed ^ (seed >> 15)
        return seed

    @ti.func
    def next_uint(self):
        # Update the state using an LCG with unsigned operations
        self.state_field[None] = (self.state_field[None] * 1664525) & ti.i64(0xffffffff)
        self.state_field[None] = (self.state_field[None] + 1013904223) & ti.i64(0xffffffff)
        return ti.cast(self.state_field[None], ti.u32)

    @ti.func
    def get_1d(self):
        return ti.cast(self.next_uint(), ti.f32) / 4294967296.0
