import taichi as ti
from taichi.math import vec3, normalize, dot, cross, sqrt, length, sin, cos, pi, vec2

from base.frame import Frame, frame_from_xz, frame_from_z
from utils.misc import fresnel



# TransportMode
RADIANCE   = 1
IMPORTANCE = 2

# BxDF Flags
BXDF_NONE = 0
BXDF_REFLECTION = 1 << 0
BXDF_TRANSMISSION = 1 << 1
BXDF_DIFFUSE = 1 << 2
BXDF_GLOSSY = 1 << 3
BXDF_SPECULAR = 1 << 4

# Composite BxDF Flags
BXDF_DIFFUSE_REFLECTION = BXDF_DIFFUSE | BXDF_REFLECTION
BXDF_DIFFUSE_TRANSMISSION = BXDF_DIFFUSE | BXDF_TRANSMISSION
BXDF_GLOSSY_REFLECTION = BXDF_GLOSSY | BXDF_REFLECTION
BXDF_GLOSSY_TRANSMISSION = BXDF_GLOSSY | BXDF_TRANSMISSION
BXDF_SPECULAR_REFLECTION = BXDF_SPECULAR | BXDF_REFLECTION
BXDF_SPECULAR_TRANSMISSION = BXDF_SPECULAR | BXDF_TRANSMISSION
BXDF_ALL = BXDF_DIFFUSE | BXDF_GLOSSY | BXDF_SPECULAR | BXDF_REFLECTION | BXDF_TRANSMISSION


PI     = pi
INV_PI = 1.0 / PI

@ti.func
def same_hemisphere(wo: vec3, wi: vec3) -> bool:
    
    ret = False
    if (wo.z * wi.z) > 0.0:
        ret = True
    return ret


@ti.func
def cos_theta(w: vec3) -> ti.f32:
    
    ret = w.z
    return ret


@ti.func
def abs_cos_theta(w: vec3) -> ti.f32:
    
    ret = ti.abs(w.z)
    return ret


@ti.func
def sqr(x: ti.f32) -> ti.f32:
    
    ret = x * x
    return ret


@ti.func
def is_inf(x: ti.f32) -> bool:
    ret = False
    if ti.abs(x) > 1e30:
        ret = True
    return ret


@ti.func
def tan2_theta(w: vec3) -> ti.f32:

    cos2 = w.z * w.z
    sin2 = ti.max(0.0, 1.0 - cos2)
    ret = 0.0
    if cos2 == 0.0:
        # infinite
        ret = 1e30
    else:
        ret = sin2 / cos2
    return ret


@ti.func
def cos2_theta(w: vec3) -> ti.f32:
    
    ret = w.z * w.z
    return ret


@ti.func
def cos_phi(w: vec3) -> ti.f32:

    denom = w.x * w.x + w.y * w.y
    ret = 0.0
    if denom > 0.0:
        ret = w.x / sqrt(denom)
    return ret


@ti.func
def sin_phi(w: vec3) -> ti.f32:

    denom = w.x * w.x + w.y * w.y
    ret = 0.0
    if denom > 0.0:
        ret = w.y / sqrt(denom)
    return ret


@ti.func
def abs_dot(a: vec3, b: vec3) -> ti.f32:
    
    dot_val = a.x * b.x + a.y * b.y + a.z * b.z
    ret = ti.abs(dot_val)
    return ret


@ti.func
def clamp(x: ti.f32, low: ti.f32, high: ti.f32) -> ti.f32:
    
    ret = x
    if x < low:
        ret = low
    elif x > high:
        ret = high
    return ret


@ti.func
def length_squared2(v: vec2) -> ti.f32:
    
    ret = v.x * v.x + v.y * v.y
    return ret


@ti.func
def lerp(t: ti.f32, v1: ti.f32, v2: ti.f32) -> ti.f32:
    
    ret = (1.0 - t) * v1 + t * v2
    return ret


@ti.func
def sample_uniform_disk_polar(u: vec2) -> vec2:

    r = sqrt(u[0])
    theta = 2.0 * PI * u[1]
    px = r * cos(theta)
    py = r * sin(theta)
    ret = vec2(px, py)
    return ret


@ti.func
def reflect(wo: vec3, n: vec3) -> vec3:
    dot_val = dot(wo, n)
    return -wo + 2.0 * dot_val * n


@ti.func
def refract(wi, n, eta):

    cosTheta_i = dot(n, wi)
    local_eta = eta
    local_n = n

    # Potentially flip interface orientation for Snell's law if cosTheta_i < 0.
    if cosTheta_i < 0.0:
        local_eta = 1.0 / local_eta
        cosTheta_i = -cosTheta_i
        local_n = -local_n

    sin2Theta_i = 1.0 - cosTheta_i * cosTheta_i
    sin2Theta_i = max(sin2Theta_i, 0.0)

    sin2Theta_t = sin2Theta_i / (local_eta * local_eta)

    valid = 1
    wt = vec3(0.0, 0.0, 0.0)
    ret_etap = local_eta

    # Check for total internal reflection.
    if sin2Theta_t >= 1.0:
        valid = 0
    else:
        cosTheta_t = ti.sqrt(1.0 - sin2Theta_t)

        inv_eta = 1.0 / local_eta
        wt = (-wi * inv_eta) + (cosTheta_i * inv_eta - cosTheta_t) * local_n
        wt = wt.normalized()

    return valid, wt, ret_etap


@ti.func
def face_forward(n: vec3, n2: vec3) -> vec3:

    ret = n
    if dot(n, n2) < 0.0:
        ret = -n
    return ret


@ti.func
def fr_dielectric(cos_theta_i: ti.f32, eta: ti.f32) -> ti.f32:
    # Clamp cosTheta_i to [-1, 1].
    c = clamp(cos_theta_i, -1.0, 1.0)
    ret = 0.0  # Final Fresnel reflectance.

    # Potentially flip interface orientation if cosTheta_i < 0.
    local_eta = eta
    local_c = c
    if c < 0.0:
        local_eta = 1.0 / eta
        local_c = -c

    sin2_i = 1.0 - (local_c * local_c)
    sin2_t = sin2_i / (local_eta * local_eta)

    # Check total internal reflection.
    if sin2_t >= 1.0:
        ret = 1.0
    else:
        # Compute cosTheta_t = sqrt(1 - sin2_t).
        cos_theta_t = ti.sqrt(1.0 - sin2_t)

        # Fresnel reflection for parallel and perpendicular polarizations.
        r_parl = (local_eta * local_c - cos_theta_t) / (local_eta * local_c + cos_theta_t)
        r_perp = (local_c - local_eta * cos_theta_t) / (local_c + local_eta * cos_theta_t)

        # Final reflectance is the average of squared magnitudes.
        ret = 0.5 * (r_parl * r_parl + r_perp * r_perp)

    return ret


@ti.func
def complex_sqr(z: vec2) -> vec2:

    return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)

@ti.func
def complex_mul(a: vec2, b: vec2) -> vec2:
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def complex_add(a: vec2, b: vec2) -> vec2:
    return a + b

@ti.func
def complex_sub(a: vec2, b: vec2) -> vec2:
    return a - b

@ti.func
def complex_conjugate(z: vec2) -> vec2:
    return vec2(z.x, -z.y)

@ti.func
def complex_div(a: vec2, b: vec2) -> vec2:
    denom = b.x * b.x + b.y * b.y
    ret = vec2(0.0, 0.0)
    if denom != 0.0:
        ret = complex_mul(a, complex_conjugate(b)) / denom
    return ret

@ti.func
def complex_norm(z: vec2) -> ti.f32:
    # Returns the squared magnitude
    return z.x * z.x + z.y * z.y

@ti.func
def complex_sqrt(z: vec2) -> vec2:
    # Computes the principal square root of a complex number.
    mag = sqrt(z.x * z.x + z.y * z.y)
    real_part = sqrt(0.5 * (mag + z.x))
    imag_part = sqrt(0.5 * (mag - z.x))
    if z.y < 0.0:
        imag_part = -imag_part
    return vec2(real_part, imag_part)




@ti.func
def fr_complex_conductor(cosTheta_i: ti.f32, eta: vec2) -> ti.f32:

    c = clamp(cosTheta_i, 0.0, 1.0)

    sin2Theta_i = 1.0 - c * c

    eta_sq = complex_sqr(eta)

    sin2Theta_i_complex = vec2(sin2Theta_i, 0.0)
    sin2Theta_t = complex_div(sin2Theta_i_complex, eta_sq)

    one_complex = vec2(1.0, 0.0)
    sub_val = complex_sub(one_complex, sin2Theta_t)
    cosTheta_t = complex_sqrt(sub_val)

    c_complex = vec2(c, 0.0)
    eta_times_c = complex_mul(eta, c_complex)
    num_parl = complex_sub(eta_times_c, cosTheta_t)
    den_parl = complex_add(eta_times_c, cosTheta_t)
    r_parl = complex_div(num_parl, den_parl)

    eta_cosTheta_t = complex_mul(eta, cosTheta_t)
    num_perp = complex_sub(c_complex, eta_cosTheta_t)
    den_perp = complex_add(c_complex, eta_cosTheta_t)
    r_perp = complex_div(num_perp, den_perp)

    result = (complex_norm(r_parl) + complex_norm(r_perp)) / 2.0
    return result

@ti.func
def fr_complex(cosTheta_i: ti.f32, eta: vec3, k: vec3) -> vec3:

    ret = vec3(0.0)
    for i in ti.static(range(3)):
        ret[i] = fr_complex_conductor(cosTheta_i, vec2(eta[i], k[i]))
    return ret


@ti.func
def sample_cosine_hemisphere(u):
    
    r = sqrt(u[0])
    theta = 2.0 * PI * u[1]
    x = r * cos(theta)
    y = r * sin(theta)
    z = sqrt(ti.max(0.0, 1.0 - x*x - y*y))
    ret = vec3(x, y, z)
    return ret


@ti.func
def cosine_hemisphere_pdf(cos_t):
    
    ret = 0.0
    if cos_t > 0.0:
        ret = cos_t * INV_PI
    return ret



@ti.dataclass
class BSDFSample:
    f:   vec3   # Spectrum or color
    wi:  vec3
    pdf: ti.f32
    flags: ti.i32
    eta:  ti.f32


@ti.dataclass
class TrowbridgeReitzDistribution:
    alpha_x: ti.f32
    alpha_y: ti.f32

    @ti.func
    def initialize(self, ax: ti.f32, ay: ti.f32):
        self.alpha_x = ax
        self.alpha_y = ay
        eff = self.effectively_smooth()
        if not eff:
            # If not effectively smooth, clamp to >= 1e-4
            self.alpha_x = ti.max(self.alpha_x, 1e-4)
            self.alpha_y = ti.max(self.alpha_y, 1e-4)

    @ti.func
    def effectively_smooth(self) -> bool:
        # return std::max(alpha_x, alpha_y) < 1e-3f
        return ti.max(self.alpha_x, self.alpha_y) < 1e-3

    @ti.func
    def D(self, wm: vec3) -> ti.f32:

        dval = 0.0
        tan2Theta = tan2_theta(wm)
        if not is_inf(tan2Theta):
            cos4Theta = sqr(cos2_theta(wm))
            if cos4Theta >= 1e-16:
                e = tan2Theta*(sqr(cos_phi(wm) / self.alpha_x) + sqr(sin_phi(wm) / self.alpha_y))
                denom = PI * self.alpha_x * self.alpha_y * cos4Theta * sqr(1.0 + e)
                dval = 1.0/denom
        return dval

    @ti.func
    def G1(self, w: vec3) -> ti.f32:

        g1v = 0.0
        lam = self.Lambda(w)
        g1v = 1.0/(1.0+ lam)
        return g1v

    @ti.func
    def Lambda(self, w: vec3) -> ti.f32:

        lam = 0.0
        tan2Th = tan2_theta(w)
        if not is_inf(tan2Th):
            alpha2 = sqr(cos_phi(w) * self.alpha_x) + sqr(sin_phi(w) * self.alpha_y)
            lam = (sqrt(1.0 + alpha2*tan2Th)- 1.0)/2.0
        return lam

    @ti.func
    def G(self, wo: vec3, wi: vec3) -> ti.f32:

        gVal = 1.0/(1.0 + self.Lambda(wo) + self.Lambda(wi))
        return gVal

    @ti.func
    def D_wo_wm(self, w: vec3, wm: vec3) -> ti.f32:

        dval = 0.0
        g1val = self.G1(w)
        denom = ti.abs(w.z)
        if denom > 0.0:
            d_wm = self.D(wm)
            ad = abs_dot(w, wm)
            dval = g1val/denom * d_wm * ad
        return dval

    @ti.func
    def pdf(self, w: vec3, wm: vec3) -> ti.f32:

        return self.D_wo_wm(w, wm)

    @ti.func
    def Sample_wm(self, w: vec3, u: vec2) -> vec3:

        wh = vec3(self.alpha_x*w.x, self.alpha_y*w.y, w.z)
        mag_wh = wh.norm()
        if mag_wh > 0.0:
            wh = wh/mag_wh
        # if wh.z < 0 => wh= -wh
        if wh.z<0.0:
            wh = -wh

        T1 = vec3(1.0, 0.0, 0.0)
        if wh.z < 0.99999:

            crossv = vec3(0.0, 0.0, 1.0).cross(wh)
            magc   = crossv.norm()
            if magc>0.0:
                T1 = crossv/magc
        T2 = wh.cross(T1)

        disk_p = sample_uniform_disk_polar(u)



        px = disk_p.x
        py = disk_p.y
        r2 = px*px + py*py

        h = sqrt(1.0 - (px*px))


        yz = py
        base = (1.0 + wh.z)*0.5



        new_py = base*(1.0 - yz) + h*yz




        p2 = px*px + new_py*new_py
        pz = 0.0
        if p2 < 1.0:
            pz = sqrt(1.0 - p2)

        nh = T1*px + T2*new_py + wh* pz


        ret = nh
        mag_nh = ret.norm()
        if mag_nh>0.0:
            ret = ret/mag_nh
        # scale x,y by alpha
        retx = self.alpha_x*ret.x
        rety = self.alpha_y*ret.y
        retz = ret.z
        if retz<1e-6:
            retz=1e-6
        final_wm = vec3(retx, rety, retz)
        # normalize again
        fmag = final_wm.norm()
        if fmag>0.0:
            final_wm= final_wm/fmag
        return final_wm

    @ti.func
    def ToString(self) -> ti.i32:


        return 0

    @ti.func
    def roughness_to_alpha(self, roughness: ti.f32) -> ti.f32:

        return sqrt(roughness)

    @ti.func
    def Regularize(self):

        if self.alpha_x < 0.3:
            self.alpha_x = clamp(2.0 * self.alpha_x, 0.1, 0.3)
        if self.alpha_y < 0.3:
            self.alpha_y = clamp(2.0 * self.alpha_y, 0.1, 0.3)



@ti.dataclass
class DiffuseBxDF:
    R: vec3

    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        # single return
        bs = BSDFSample(vec3(0), vec3(0), 0.0, 0, 1.0)
        canReflect = (sample_flags & BXDF_REFLECTION) != 0
        if canReflect:
            # Cosine hemisphere
            hemi = sample_cosine_hemisphere(u2)
            wi = hemi
            if wo.z < 0.0:
                wi.z = -wi.z
            pdfv = cosine_hemisphere_pdf(ti.abs(wi.z))
            fv   = self.R * INV_PI

            bs.f     = fv
            bs.wi    = wi
            bs.pdf   = pdfv
            bs.flags = (BXDF_DIFFUSE | BXDF_REFLECTION)
        return bs

    @ti.func
    def f(self, wo: vec3, wi: vec3, mode: ti.i32) -> vec3:
        fval = vec3(0.0)
        if same_hemisphere(wo, wi):
            fval = self.R * INV_PI
        return fval

    @ti.func
    def pdf(self, wo: vec3, wi: vec3, mode: ti.i32, sample_flags: ti.i32) -> ti.f32:
        pdfv = 0.0
        canReflect = (sample_flags & BXDF_REFLECTION) != 0
        if canReflect and same_hemisphere(wo, wi):
            pdfv = cosine_hemisphere_pdf(ti.abs(wi.z))
        return pdfv

    @ti.func
    def flags(self) -> ti.i32:

        flag_val = 0
        # If R has any non-zero component
        if self.R.max() > 0.0:
            flag_val = BXDF_DIFFUSE | BXDF_REFLECTION
        return flag_val



@ti.dataclass
class DiffuseTransmissionBxDF:
    R: vec3
    T: vec3

    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        # single return
        bs = BSDFSample(vec3(0), vec3(0), 0.0, 0, 1.0)
        # get reflection or transmission "max" to pick
        pr = self.R.max()
        pt = self.T.max()
        refOk = (sample_flags & BXDF_REFLECTION) != 0
        traOk = (sample_flags & BXDF_TRANSMISSION) != 0
        if not refOk:
            pr = 0.0
        if not traOk:
            pt = 0.0
        sump = pr + pt
        if sump > 0.0:
            # pick reflection or transmission
            branch = 0
            if uc < pr/sump:
                branch = 1
            else:
                branch = 2
            if branch == 1:
                # reflection
                wi = sample_cosine_hemisphere(u2)
                if wo.z < 0.0:
                    wi.z = -wi.z
                pdfv = cosine_hemisphere_pdf(ti.abs(wi.z)) * (pr / sump)
                fval = self.f(wo, wi, mode)
                bs.f   = fval
                bs.wi  = wi
                bs.pdf = pdfv
                bs.flags = (BXDF_DIFFUSE | BXDF_REFLECTION)
            else:
                # transmission
                wi = sample_cosine_hemisphere(u2)
                if wo.z > 0.0:
                    wi.z = -wi.z
                pdfv = cosine_hemisphere_pdf(ti.abs(wi.z)) * (pt / sump)
                fval = self.f(wo, wi, mode)
                bs.f   = fval
                bs.wi  = wi
                bs.pdf = pdfv
                bs.flags = (BXDF_DIFFUSE | BXDF_TRANSMISSION)
        return bs

    @ti.func
    def f(self, wo: vec3, wi: vec3, mode: ti.i32) -> vec3:
        fval = vec3(0)
        if same_hemisphere(wo, wi):
            fval = self.R * INV_PI
        else:
            fval = self.T * INV_PI
        return fval

    @ti.func
    def pdf(self, wo: vec3, wi: vec3, mode: ti.i32, sample_flags: ti.i32) -> ti.f32:
        pdfv = 0.0
        refOk = (sample_flags & BXDF_REFLECTION) != 0
        traOk = (sample_flags & BXDF_TRANSMISSION) != 0
        pr = self.R.max() if refOk else 0.0
        pt = self.T.max() if traOk else 0.0
        sump = pr + pt
        if sump > 0.0:
            if same_hemisphere(wo, wi):
                pdfv = (pr/sump) * cosine_hemisphere_pdf(ti.abs(wi.z))
            else:
                pdfv = (pt/sump) * cosine_hemisphere_pdf(ti.abs(wi.z))
        return pdfv

    @ti.func
    def flags(self) -> ti.i32:

        flag_val = 0
        anyR = (self.R.max() > 0.0)
        anyT = (self.T.max() > 0.0)
        if anyR or anyT:
            # always Diffuse
            flag_val |= BXDF_DIFFUSE
            if anyR:
                flag_val |= BXDF_REFLECTION
            if anyT:
                flag_val |= BXDF_TRANSMISSION
        return flag_val




@ti.dataclass
class DielectricBxDF:
    eta: ti.f32         # index of refraction for the dielectric
    color: vec3         # tint (if any)
    mf_distrib: TrowbridgeReitzDistribution

    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        # Choose the smooth vs. rough branch based on the microfacet distribution.
        bs = BSDFSample(vec3(0.0), vec3(0.0), 0.0, 0, 1.0)
        if self.eta == 1.0 or self.mf_distrib.effectively_smooth():
            bs = self.sample_f_Smooth(wo, uc, u2, mode, sample_flags)
        else:
            bs = self.sample_f_Rough(wo, uc, u2, mode, sample_flags)
        return bs

    @ti.func
    def alt_sample_f_Smooth(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        # This is the perfect specular case.
        out = BSDFSample(vec3(0.0), vec3(0.0), 0.0, 0, 1.0)
        R = fr_dielectric(cos_theta(wo), self.eta)
        T = 1.0 - R

        pr = R
        pt = T
        if (sample_flags & BXDF_REFLECTION) == 0:
            pr = 0.0
        if (sample_flags & BXDF_TRANSMISSION) == 0:
            pt = 0.0
        s = pr + pt
        pick = 0
        if pr > 0 and pt > 0:
            if uc < (pr / s):
                pick = 1
            else:
                pick = 2

        if pick == 1:

            wi = vec3(-wo.x, -wo.y, wo.z)
            out.wi = wi
            out.pdf = pr / s
            cos_i = abs_cos_theta(wi)
            fac = 0.0
            if cos_i != 0.0:
                fac = R / cos_i
            out.f = self.color * fac
            out.flags = BXDF_SPECULAR | BXDF_REFLECTION
            out.eta = self.eta
        elif pick == 2:
            # Transmission branch: compute refracted ray using a surface normal of (0,0,1)
            valid, wt, etap = refract(wo, vec3(0.0, 0.0, 1.0), self.eta)
            if valid == 0:
                # Total internal reflection fallback: return perfect reflection.
                print("RARE TIR!")
                wi2 = vec3(-wo.x, -wo.y, wo.z)
                out.wi = wi2
                out.f = self.color
                out.pdf = 1.0
                out.flags = BXDF_SPECULAR | BXDF_REFLECTION
                out.eta = 1.0
            else:
                out.wi = wt
                out.pdf = pt / s
                cos_i = abs_cos_theta(wt)
                fac = 0.0
                if cos_i != 0.0:
                    fac = T / cos_i
                if mode == RADIANCE:
                    fac /= (etap * etap)
                out.f = self.color * fac
                out.flags = BXDF_SPECULAR | BXDF_TRANSMISSION
                out.eta = etap
        return out

    @ti.func
    def sample_f_Smooth(self, wo, uc, u, mode, sample_flags):
        # Initialize output.
        bs = BSDFSample()

        # Compute Fresnel reflectance and transmission.
        R, cos_t, eta_it, eta_ti = fresnel(cos_theta(wo), self.eta)
        T = 1.0 - R

        # Compute probabilities for reflection and transmission.
        pr = R if (sample_flags & BXDF_REFLECTION) != 0 else 0.0
        pt = T if (sample_flags & BXDF_TRANSMISSION) != 0 else 0.0

        # Determine whether to reflect or refract.
        if pr > 0 and pt > 0:
            if uc < pr / (pr + pt):
                # Reflection branch.
                wi = vec3(-wo.x, -wo.y, wo.z)
                fr = R / abs_cos_theta(wi)
                bs.f = fr * self.color
                bs.wi = wi
                bs.eta = 1.0
                bs.pdf = pr / (pr + pt)
                bs.flags = BXDF_SPECULAR | BXDF_REFLECTION
            else:
                # Transmission branch.
                n = vec3(0.0, 0.0, 1.0) if cos_theta(wo) > 0 else vec3(0.0, 0.0, -1.0)
                d = -wo  # Incident direction.

                # Use eta_it and eta_ti from Fresnel calculation.
                eta_i = 1.0 if eta_it == self.eta else self.eta
                eta_t = self.eta if eta_it == self.eta else 1.0

                # Compute refraction.
                n_dot_i = d.dot(n)
                eta = eta_i / eta_t
                sin2_theta_i = max(0.0, 1 - n_dot_i ** 2)
                sin2_theta_t = sin2_theta_i * eta ** 2

                if sin2_theta_t <= 1.0:
                    # Refraction is valid.
                    wi = (eta * d + (eta * n_dot_i - cos_t) * n).normalized()
                    ft = T / abs_cos_theta(wi)
                    if mode:
                        ft /= (eta_t ** 2)
                    bs.f = ft * self.color
                    bs.wi = wi
                    bs.pdf = pt / (pr + pt)
                    bs.eta = eta_t
                    bs.flags = BXDF_SPECULAR | BXDF_TRANSMISSION
                else:
                    # Total internal reflection fallback.
                    wi = vec3(-wo.x, -wo.y, wo.z)
                    bs.f = self.color
                    bs.wi = wi
                    bs.pdf = 1.0
                    bs.eta = 1.0
                    bs.flags = BXDF_SPECULAR | BXDF_REFLECTION
        else:
            # Handle cases where only reflection or transmission is allowed.
            if pr > 0:
                wi = vec3(-wo.x, -wo.y, wo.z)
                fr = R / abs_cos_theta(wi)
                bs.f = fr * self.color
                bs.wi = wi
                bs.eta = 1.0
                bs.pdf = 1.0
                bs.flags = BXDF_SPECULAR | BXDF_REFLECTION
            elif pt > 0:
                n = vec3(0.0, 0.0, 1.0) if cos_theta(wo) > 0 else vec3(0.0, 0.0, -1.0)
                d = -wo  # Incident direction.

                # Use eta_it and eta_ti from Fresnel calculation.
                eta_i = 1.0 if eta_it == self.eta else self.eta
                eta_t = self.eta if eta_it == self.eta else 1.0

                # Compute refraction.
                n_dot_i = d.dot(n)
                eta = eta_i / eta_t
                sin2_theta_i = max(0.0, 1 - n_dot_i ** 2)
                sin2_theta_t = sin2_theta_i * eta ** 2

                if sin2_theta_t <= 1.0:
                    # Refraction is valid.
                    wi = (eta * d + (eta * n_dot_i - cos_t) * n).normalized()
                    ft = T / abs_cos_theta(wi)
                    if mode:
                        ft /= (eta_t ** 2)
                    bs.f = ft * self.color
                    bs.wi = wi
                    bs.pdf = 1.0
                    bs.eta = eta_t
                    bs.flags = BXDF_SPECULAR | BXDF_TRANSMISSION
                else:
                    # Total internal reflection fallback.
                    wi = vec3(-wo.x, -wo.y, wo.z)
                    bs.f = self.color
                    bs.wi = wi
                    bs.pdf = 1.0
                    bs.eta = 1.0
                    bs.flags = BXDF_SPECULAR | BXDF_REFLECTION

        return bs

    @ti.func
    def sample_f_Rough(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        out = BSDFSample(vec3(0.0), vec3(0.0), 0.0, 0, 1.0)
        # Sample a microfacet normal wm using the Trowbridge-Reitz distribution.
        wm = self.mf_distrib.Sample_wm(wo, u2)
        Rval = fr_dielectric(dot(wo, wm), self.eta)
        Tval = 1.0 - Rval

        pr = 0.0
        pt = 0.0
        if (sample_flags & BXDF_REFLECTION) != 0:
            pr = Rval
        if (sample_flags & BXDF_TRANSMISSION) != 0:
            pt = Tval
        s = pr + pt
        if s > 0.0:
            pick = 0
            if uc < (pr / s):
                pick = 1
            else:
                pick = 2

            if pick == 1:
                # Reflection branch:
                wi_r = reflect(wo, wm)
                if same_hemisphere(wo, wi_r):
                    cos_i = abs_cos_theta(wi_r)
                    cos_o = abs_cos_theta(wo)
                    if cos_i != 0.0 and cos_o != 0.0:
                        denom = 4.0 * ti.abs(cos_theta(wi_r)) * ti.abs(cos_theta(wo))
                        dval = self.mf_distrib.D(wm)
                        gval = self.mf_distrib.G(wo, wi_r)
                        fF = Rval
                        f_brdf = 0.0
                        if denom != 0.0:
                            f_brdf = (dval * gval * fF) / denom
                        pdfv = self.mf_distrib.pdf(wo, wm) / (4.0 * ti.abs(dot(wo, wm))) * (pr / s)
                        out.f = self.color * vec3(f_brdf)
                        out.wi = wi_r
                        out.pdf = pdfv
                        out.flags = BXDF_GLOSSY | BXDF_REFLECTION
            else:
                # Transmission branch:
                valid, wi_t, etap = refract(wo, wm, self.eta)
                if valid != 0:
                    if (not same_hemisphere(wo, wi_t)) and (wi_t.z != 0.0):
                        denom = (dot(wi_t, wm) + dot(wo, wm) / etap)**2
                        dwm_dwi = ti.abs(dot(wi_t, wm)) / denom
                        pdfv = self.mf_distrib.pdf(wo, wm) * dwm_dwi * (pt / s)
                        dval = self.mf_distrib.D(wm)
                        gval = self.mf_distrib.G(wo, wi_t)
                        ft = Tval * dval * gval * ti.abs(dot(wi_t, wm) * dot(wo, wm) /
                               (cos_theta(wi_t) * cos_theta(wo) * denom))
                        if mode == RADIANCE:
                            ft /= (etap * etap)
                        out.f = self.color * vec3(ft)
                        out.wi = wi_t
                        out.pdf = pdfv
                        out.flags = BXDF_GLOSSY | BXDF_TRANSMISSION
        return out

    @ti.func
    def f(self, wo: vec3, wi: vec3, mode: ti.i32) -> vec3:


        fval = vec3(0.0)
        if not (self.eta == 1.0 or self.mf_distrib.effectively_smooth()):
            cos_o = cos_theta(wo)
            cos_i = cos_theta(wi)
            reflect_branch = (cos_i * cos_o) > 0.0
            etap = 1.0
            if not reflect_branch:
                if cos_o > 0.0:
                    etap = self.eta
                else:
                    etap = 1.0 / self.eta
            wm = (wi * etap) + wo
            if wm.x != 0 or wm.y != 0 or wm.z != 0:
                wmN = face_forward(normalize(wm), vec3(0, 0, 1))
                # Discard backfacing microfacets.
                if dot(wmN, wi) * cos_i >= 0 and dot(wmN, wo) * cos_o >= 0:
                    F = fr_dielectric(dot(wo, wmN), self.eta)
                    if reflect_branch:
                        denom = 4.0 * ti.abs(cos_i) * ti.abs(cos_o)
                        if denom != 0.0:
                            _fval = self.mf_distrib.D(wmN) * self.mf_distrib.G(wo, wi) * F / denom
                            fval = vec3(_fval)
                    else:
                        denom = (dot(wi, wmN) + dot(wo, wmN) / etap)**2 * cos_i * cos_o
                        if denom != 0.0:
                            ft = self.mf_distrib.D(wmN) * (1.0 - F) * self.mf_distrib.G(wo, wi) * ti.abs(dot(wi, wmN) * dot(wo, wmN) / denom)
                            if mode == RADIANCE:
                                ft /= (etap * etap)
                            fval = vec3(ft)
        return fval

    @ti.func
    def pdf(self, wo: vec3, wi: vec3, mode: ti.i32, sample_flags: ti.i32) -> ti.f32:
        pdfv = 0.0
        if self.eta == 1.0 or self.mf_distrib.effectively_smooth():
            pdfv = 0.0
        else:
            cos_o = cos_theta(wo)
            cos_i = cos_theta(wi)
            reflect_branch = (cos_o * cos_i) > 0.0
            etap = 1.0
            if not reflect_branch:
                if cos_o > 0:
                    etap = self.eta
                else:
                    etap = 1.0 / self.eta
            wm = (wi * etap) + wo
            if wm.x != 0 or wm.y != 0 or wm.z != 0:
                wmN = face_forward(normalize(wm), vec3(0, 0, 1))
                if dot(wmN, wi) * cos_i > 0 and dot(wmN, wo) * cos_o > 0:
                    R = fr_dielectric(dot(wo, wmN), self.eta)
                    T = 1.0 - R
                    pr = R
                    pt = T
                    if (sample_flags & BXDF_REFLECTION) == 0:
                        pr = 0.0
                    if (sample_flags & BXDF_TRANSMISSION) == 0:
                        pt = 0.0
                    if pr == 0 and pt == 0:
                        pdfv = 0.0
                    else:
                        sum_p = pr + pt
                        if reflect_branch:
                            valD = self.mf_distrib.pdf(wo, wmN) / (4.0 * ti.abs(dot(wo, wmN)))
                            pdfv = valD * (pr / sum_p)
                        else:
                            denom = (dot(wi, wmN) + (dot(wo, wmN) / etap))
                            denom = denom * denom
                            dwm = ti.abs(dot(wi, wmN)) / denom
                            valD = self.mf_distrib.pdf(wo, wmN)
                            pdfv = valD * dwm * (pt / sum_p)
        return pdfv

    @ti.func
    def flags(self) -> ti.i32:
        flag_val = 0
        # If η == 1, then use only transmission; otherwise reflection + transmission.
        if self.eta == 1.0:
            flag_val = BXDF_TRANSMISSION
        else:
            flag_val = BXDF_REFLECTION | BXDF_TRANSMISSION

        # Then add the specular/glossy flag based on the microfacet roughness.
        if self.mf_distrib.effectively_smooth():
            flag_val |= BXDF_SPECULAR
        else:
            flag_val |= BXDF_GLOSSY

        return flag_val



@ti.dataclass
class ConductorBxDF:
    eta: vec3
    k:   vec3
    mf_distrib: TrowbridgeReitzDistribution


    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        bs = BSDFSample(vec3(0.0), vec3(0.0), 0.0, 0, 1.0)
        # Only reflection for conductors.
        if (sample_flags & BXDF_REFLECTION) != 0:
            if self.mf_distrib.effectively_smooth():
                # Smooth (perfect mirror) case

                wi = vec3(-wo.x, -wo.y, wo.z)
                cosi = abs_cos_theta(wi)
                if cosi != 0.0:
                    # Compute Fresnel factor F = FrComplex(|cosTheta|, eta, k)/|cosTheta|
                    F = fr_complex(cosi, self.eta, self.k) * (1.0 / cosi)
                    bs.f = F
                    bs.wi = wi
                    bs.pdf = 1.0
                    bs.flags = BXDF_SPECULAR | BXDF_REFLECTION
            else:
                # Rough conductor: sample a microfacet normal
                wm = self.mf_distrib.Sample_wm(wo, u2)
                # Use the sampled microfacet normal to compute the reflected direction.
                wi = reflect(wo, wm)
                if same_hemisphere(wo, wi):
                    cos_i = abs_cos_theta(wi)
                    cos_o = abs_cos_theta(wo)
                    if cos_i != 0.0 and cos_o != 0.0:
                        # Compute microfacet distribution terms
                        dval = self.mf_distrib.D(wm)
                        gval = self.mf_distrib.G(wo, wi)
                        # Use the sampled microfacet normal for the Fresnel term.
                        F = fr_complex(ti.abs(dot(wo, wm)), self.eta, self.k)
                        denom = 4.0 * cos_i * cos_o
                        f_brdf = vec3(0.0)
                        if denom != 0.0:

                            _f_brdf = (dval * gval / denom)
                            f_brdf = _f_brdf * F
                        pdfv = self.mf_distrib.pdf(wo, wm) / (4.0 * ti.abs(dot(wo, wm)))
                        bs.f = f_brdf
                        bs.wi = wi
                        bs.pdf = pdfv
                        bs.flags = BXDF_GLOSSY | BXDF_REFLECTION
        return bs

    @ti.func
    def f(self, wo: vec3, wi: vec3, mode: ti.i32) -> vec3:
        fval = vec3(0.0)
        if same_hemisphere(wo, wi) and not self.mf_distrib.effectively_smooth():
            cos_o = abs_cos_theta(wo)
            cos_i = abs_cos_theta(wi)
            if cos_o != 0.0 and cos_i != 0.0:
                wm = wo + wi
                if (wm.x != 0.0 or wm.y != 0.0 or wm.z != 0.0):
                    wmN = face_forward(normalize(wm), vec3(0, 0, 1))
                    F = fr_complex(ti.abs(dot(wo, wmN)), self.eta, self.k)
                    dval = self.mf_distrib.D(wmN)
                    gval = self.mf_distrib.G(wo, wi)
                    denom = 4.0 * cos_o * cos_i
                    if denom != 0.0:
                        factor = dval * gval / denom
                        fval = factor * F
        return fval

    @ti.func
    def pdf(self, wo: vec3, wi: vec3, mode: ti.i32, sample_flags: ti.i32) -> ti.f32:
        pdfv = 0.0
        if (sample_flags & BXDF_REFLECTION) != 0 and same_hemisphere(wo, wi):
            if not self.mf_distrib.effectively_smooth():
                wm = wo + wi
                if wm.x != 0.0 or wm.y != 0.0 or wm.z != 0.0:
                    wmN = face_forward(normalize(wm), vec3(0, 0, 1))
                    valD = self.mf_distrib.pdf(wo, wmN)
                    denom = 4.0 * ti.abs(dot(wo, wmN))
                    if denom != 0.0:
                        pdfv = valD / denom
        return pdfv

    @ti.func
    def flags(self) -> ti.i32:
        flag_val = 0

        if self.mf_distrib.effectively_smooth():
            flag_val |= BXDF_SPECULAR
            flag_val |= BXDF_REFLECTION
        else:
            flag_val |= BXDF_GLOSSY
            flag_val |= BXDF_REFLECTION
        return flag_val


@ti.dataclass
class BSDF:


    type: ti.i32

    diffuse: DiffuseBxDF
    diffuse_transmission: DiffuseTransmissionBxDF
    dielectric: DielectricBxDF
    conductor: ConductorBxDF

    frame: Frame

    @ti.func
    def to_local(self, v: vec3) -> vec3:
        return self.frame.to_local(v)

    @ti.func
    def from_local(self, v: vec3) -> vec3:
        return self.frame.from_local(v)

    @ti.func
    def init_frame(self, normal):
        self.frame = frame_from_z(normal)

    @ti.func
    def add_diffuse(self, R):
        self.diffuse.R = R
        self.type = 0

    @ti.func
    def add_transmission(self, R, T):
        self.diffuse_transmission.R = R
        self.diffuse_transmission.T = T
        self.type = 1

    @ti.func
    def add_dielectric(self, eta, color, uroughness, vroughness):
        # print(eta, color, uroughness, vroughness)
        self.dielectric.eta = eta
        self.dielectric.color = color
        alpha_x = self.dielectric.mf_distrib.roughness_to_alpha(uroughness)
        alpha_y = self.dielectric.mf_distrib.roughness_to_alpha(vroughness)
        print(alpha_x, alpha_y)
        self.dielectric.mf_distrib.initialize(alpha_x, alpha_y)
        self.type = 2

    @ti.func
    def add_conductor(self, eta, k, uroughness, vroughness):
        # print("Adding conductor", eta, k, uroughness, vroughness)
        self.conductor.eta = eta
        self.conductor.k = k
        alpha_x = self.conductor.mf_distrib.roughness_to_alpha(uroughness)
        alpha_y = self.conductor.mf_distrib.roughness_to_alpha(vroughness)
        self.conductor.mf_distrib.initialize(alpha_x, alpha_y)
        self.type = 3

    @ti.func
    def f(self, wo_world, wi_world, mode=1):
        wo = self.to_local(wo_world)
        wi = self.to_local(wi_world)
        result = vec3(0.0)
        if wo.z != 0:
            if self.type == 0:
                result = self.diffuse.f(wo, wi, mode)
            elif self.type == 1:
                result = self.diffuse_transmission.f(wo, wi, mode)
            elif self.type == 2:
                result = self.dielectric.f(wo, wi, mode)
            elif self.type == 3:
                result = self.conductor.f(wo, wi, mode)
        return result

    @ti.func
    def sample_f(self, wo_world, u, u2, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()
        bxdf_sample = bs

        wo = self.to_local(wo_world)

        if self.type == 0:
            if wo.z != 0 and (self.diffuse.flags() & sample_flags != 0):
                bxdf_sample = self.diffuse.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == 1:
            if wo.z != 0 and (self.diffuse_transmission.flags() & sample_flags != 0):
                bxdf_sample = self.diffuse_transmission.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == 2:
            if wo.z != 0 and (self.dielectric.flags() & sample_flags != 0):
                bxdf_sample = self.dielectric.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == 3:
            if wo.z != 0 and (self.conductor.flags() & sample_flags != 0):
                bxdf_sample = self.conductor.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)

        if not bs.f.max() > 0 and bxdf_sample.pdf != 0 and bxdf_sample.wi.z != 0:
            bs.f = bxdf_sample.f
            bs.wi = bxdf_sample.wi
            bs.pdf = bxdf_sample.pdf
            bs.flags = bxdf_sample.flags

        return bs

    @ti.func
    def pdf(self, wo_world, wi_world, mode=1, sample_flags=BXDF_ALL):
        wo = self.to_local(wo_world)
        wi = self.to_local(wi_world)
        result = 0.0
        if wo.z != 0:
            if self.type == 0:
                result = self.diffuse.pdf(wo, wi, mode, sample_flags)
            elif self.type == 1:
                result = self.diffuse_transmission.pdf(wo, wi, mode, sample_flags)
            elif self.type == 2:
                result = self.dielectric.pdf(wo, wi, mode, sample_flags)
            elif self.type == 3:
                result = self.conductor.pdf(wo, wi, mode, sample_flags)
        return result

    @ti.func
    def flags(self):
        flag = BXDF_NONE
        if self.type == 0:
            flag = self.diffuse.flags()
        elif self.type == 1:
            flag = self.diffuse_transmission.flags()
        elif self.type == 2:
            flag = self.dielectric.flags()
        elif self.type == 3:
            flag = self.conductor.flags()

        return flag
