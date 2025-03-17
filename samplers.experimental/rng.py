import taichi as ti

@ti.dataclass
class RNG:
    state: ti.u64
    inc: ti.u64

    @ti.func
    def pcg32_uniform_uint32(self) -> ti.u32:
        oldstate = self.state
        self.state = oldstate * ti.u64(0x5851f42d4c957f2d) + self.inc
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = ti.u32(oldstate >> 59) & 31
        result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
        return result & ti.u64(0xFFFFFFFF)

    @ti.func
    def uniform_uint64(self) -> ti.u64:
        v0 = ti.u64(self.pcg32_uniform_uint32())
        v1 = ti.u64(self.pcg32_uniform_uint32())
        return (v0 << 32) | v1

    @ti.func
    def uniform_float(self) -> ti.f32:
        ret = ti.min(1.0 - 1e-8, ti.cast(self.pcg32_uniform_uint32(), ti.f32) * 2.3283064365386963e-10)
        return ret

    @ti.func
    def uniform_double(self) -> ti.f64:
        ret = ti.min(1.0 - 1e-16, ti.cast(self.uniform_uint64(), ti.f64) * 5.421010862427522e-20)
        return ret

    @ti.func
    def set_sequence(self, seq_idx: ti.u64, seed: ti.u64):
        self.state = 0
        self.inc = (seq_idx << 1) | ti.u64(1)
        self.pcg32_uniform_uint32()  # First random step to mix bits
        self.state = (self.state + seed) * ti.u64(6364136223846793005) + self.inc
        self.pcg32_uniform_uint32()  # Second step ensures sequence starts correctly

    @ti.func
    def advance(self, delta: ti.i64):
        cur_mult = ti.u64(0x5851f42d4c957f2d)
        cur_plus = self.inc
        acc_mult = ti.u64(1)
        acc_plus = ti.u64(0)
        udelta = ti.u64(delta)
        while udelta > 0:
            if udelta & ti.u64(1):
                acc_mult *= cur_mult
                acc_plus = acc_plus * cur_mult + cur_plus
            cur_plus = (cur_mult + ti.u64(1)) * cur_plus
            cur_mult *= cur_mult
            udelta >>= 1
        self.state = acc_mult * self.state + acc_plus

    @ti.func
    def get_2d(self):
        return ti.Vector([self.uniform_float(), self.uniform_float()])
    
    @ti.func
    def get_1d(self):
        return self.uniform_float()
