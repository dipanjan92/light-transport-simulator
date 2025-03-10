import taichi as ti

@ti.dataclass
class RNG:
    state_field: ti.u64

    @ti.func
    def next_uint(self) -> ti.u64:
        # Standard 32-bit LCG logic, but stored in a 64-bit field
        # We'll mask out only the lower 32 bits.
        self.state_field = (ti.u64(1664525) * self.state_field + ti.u64(1013904223)) & ti.u64(0xffffffff)
        return self.state_field

    @ti.func
    def uniform_float(self) -> ti.f32:
        # Convert the 32-bit random integer to a float in [0,1)
        return ti.cast(self.next_uint(), ti.f32) / 4294967296.0

    @ti.func
    def get_2d(self):
        return ti.Vector([self.uniform_float(), self.uniform_float()])
    
    @ti.func
    def get_1d(self):
        return self.uniform_float()
