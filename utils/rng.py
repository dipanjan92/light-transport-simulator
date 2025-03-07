import taichi as ti

@ti.data_oriented
class RNG:
    def __init__(self, seed: ti.i32):
        # The seed/state is stored as an integer.
        self.state = seed

    @ti.func
    def next_uint(self) -> ti.u32:
        # A simple LCG: state = (a * state + c) mod 2^32.
        # Constants chosen are similar to those often used.
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return self.state

    @ti.func
    def uniform_float(self) -> ti.f32:
        # Return a float in [0, 1)
        return ti.cast(self.next_uint(), ti.f32) / 4294967296.0

