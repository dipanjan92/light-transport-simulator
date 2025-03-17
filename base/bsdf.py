"""
This module implements various BSDF and BxDF models for simulating light transport in rendering.
It includes implementations for diffuse reflection, diffuse transmission, dielectric materials,
and conductors. These models are based on microfacet theory and are used to evaluate and sample
the Bidirectional Scattering Distribution Function (BSDF) in a physically-based rendering system.
"""

import taichi as ti
from taichi.math import vec3, normalize, dot, cross, sqrt, length, sin, cos, pi, vec2

from base.frame import Frame, frame_from_xz, frame_from_z

from utils.vecmath import *
from utils.scattering import *
from utils.complex import *
from utils.constants import *


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



@ti.dataclass
class BSDFSample:
    """
    Data class representing a BSDF sample.
    
    Attributes:
        f (vec3): The sampled spectral value (color).
        wi (vec3): The sampled incident direction.
        pdf (ti.f32): The probability density of the sample.
        flags (ti.i32): Flags indicating the type of scattering.
        eta (ti.f32): The index of refraction at the sample.
    """
    f:   vec3   # Spectrum or color
    wi:  vec3
    pdf: ti.f32
    flags: ti.i32
    eta:  ti.f32


@ti.dataclass
class TrowbridgeReitzDistribution:
    """
    Represents the Trowbridge-Reitz microfacet distribution used for modeling rough surface reflection.
    """
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
    """
    Represents a diffuse (Lambertian) BRDF for reflection.
    """
    R: vec3

    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        """
        Samples the BRDF to generate a BSDFSample for diffuse reflection using cosine-weighted hemisphere sampling.

        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number for branch selection.
            u2 (vec2): A 2D random sample for hemisphere sampling.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).
            sample_flags (ti.i32): Flags specifying which components to sample (e.g., reflection).

        Returns:
            BSDFSample: The sampled BSDF value including spectral value, incident direction, PDF, and flags.
        """
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
        """
        Computes the diffuse BRDF value for given outgoing and incident directions.

        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).

        Returns:
            vec3: The computed BRDF value.
        """
        fval = vec3(0.0)
        if same_hemisphere(wo, wi):
            fval = self.R * INV_PI
        return fval

    @ti.func
    def pdf(self, wo: vec3, wi: vec3, mode: ti.i32, sample_flags: ti.i32) -> ti.f32:
        """
        Computes the probability density function (PDF) for the diffuse BRDF given outgoing and incident directions.

        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag.
            sample_flags (ti.i32): Flags indicating which components are being sampled.

        Returns:
            ti.f32: The probability density of the sample.
        """
        pdfv = 0.0
        canReflect = (sample_flags & BXDF_REFLECTION) != 0
        if canReflect and same_hemisphere(wo, wi):
            pdfv = cosine_hemisphere_pdf(ti.abs(wi.z))
        return pdfv

    @ti.func
    def flags(self) -> ti.i32:
        """
        Returns the BxDF flags for the diffuse BRDF based on its reflectance properties.

        Returns:
            ti.i32: A flag indicating that the BRDF is diffuse and reflective.
        """

        flag_val = 0
        # If R has any non-zero component
        if self.R.max() > 0.0:
            flag_val = BXDF_DIFFUSE | BXDF_REFLECTION
        return flag_val



@ti.dataclass
class DiffuseTransmissionBxDF:
    """
    Represents a diffuse transmission BxDF that models both reflection and transmission in diffuse surfaces.
    This class uses reflectance (R) and transmittance (T) properties to probabilistically sample either a reflection or transmission event.
    """
    R: vec3
    T: vec3

    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        """
        Samples the BSDF for diffuse transmission by choosing between reflection and transmission based on the maximum values of R and T.
        The method uses a random number (uc) to decide the branch and returns a BSDFSample with the appropriate parameters.
        
        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number for branch selection.
            u2 (vec2): A 2D random sample used for hemisphere sampling.
            mode (ti.i32): Mode flag indicating the transport mode.
            sample_flags (ti.i32): Flags specifying allowed scattering components.
        
        Returns:
            BSDFSample: The resulting sample containing the spectral value, incoming direction, PDF, and flags.
        """
        # Create a default BSDFSample with zeroed values.
        bs = BSDFSample(vec3(0), vec3(0), 0.0, 0, 1.0)
        # Determine the maximum reflectance and transmittance values from R and T.
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
            # Probabilistically select between reflection (branch 1) and transmission (branch 2) based on the ratio of pr to pt.
            branch = 0
            if uc < pr/sump:
                branch = 1
            else:
                branch = 2
            if branch == 1:
                # Reflection branch:
                # Sample a cosine-weighted hemisphere for reflection. Adjust the sign of wi.z based on wo.z.
                wi = sample_cosine_hemisphere(u2)
                if wo.z < 0.0:
                    wi.z = -wi.z
                pdfv = cosine_hemisphere_pdf(ti.abs(wi.z)) * (pr / sump)
                fval = self.f(wo, wi, mode)
                bs.f = fval
                bs.wi = wi
                bs.pdf = pdfv
                bs.flags = (BXDF_DIFFUSE | BXDF_REFLECTION)
            else:
                # Transmission branch:
                # Sample a cosine-weighted hemisphere for transmission. Invert wi.z if necessary for proper transmission direction.
                wi = sample_cosine_hemisphere(u2)
                if wo.z > 0.0:
                    wi.z = -wi.z
                pdfv = cosine_hemisphere_pdf(ti.abs(wi.z)) * (pt / sump)
                fval = self.f(wo, wi, mode)
                bs.f = fval
                bs.wi = wi
                bs.pdf = pdfv
                bs.flags = (BXDF_DIFFUSE | BXDF_TRANSMISSION)
        return bs

    @ti.func
    def f(self, wo: vec3, wi: vec3, mode: ti.i32) -> vec3:
        """
        Computes the BSDF value for diffuse transmission given outgoing and incident directions.
        Uses reflectance R if the directions are in the same hemisphere (reflection) and transmittance T otherwise (transmission).
        
        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag.
        
        Returns:
            vec3: The computed BSDF value.
        """
        fval = vec3(0)
        if same_hemisphere(wo, wi):
            fval = self.R * INV_PI
        else:
            fval = self.T * INV_PI
        return fval

    @ti.func
    def pdf(self, wo: vec3, wi: vec3, mode: ti.i32, sample_flags: ti.i32) -> ti.f32:
        """
        Computes the probability density function (PDF) for the diffuse transmission BSDF.
        The PDF is weighted by the relative contributions of reflectance and transmittance based on the sampling branch.
        
        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag.
            sample_flags (ti.i32): Flags indicating which scattering components are allowed.
        
        Returns:
            ti.f32: The computed probability density.
        """
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
        """
        Returns the combined BxDF flags for diffuse transmission based on non-zero reflectance and transmittance values.
        The flags indicate that the BSDF is diffuse and may support reflection and/or transmission.
        
        Returns:
            ti.i32: The combined flags for the BxDF.
        """
        flag_val = 0
        anyR = (self.R.max() > 0.0)
        anyT = (self.T.max() > 0.0)
        if anyR or anyT:
            # Always mark as diffuse.
            flag_val |= BXDF_DIFFUSE
            if anyR:
                flag_val |= BXDF_REFLECTION
            if anyT:
                flag_val |= BXDF_TRANSMISSION
        return flag_val



@ti.dataclass
class DielectricBxDF:
    """
    Represents the BSDF for a dielectric material, handling both reflection and transmission,
    with separate branches for smooth (specular) and rough surfaces.
    """
    eta: ti.f32         # index of refraction for the dielectric
    color: vec3         # tint (if any)
    mf_distrib: TrowbridgeReitzDistribution

    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        """
        Samples the BSDF to generate a BSDFSample for dielectric materials by choosing between smooth and rough branches based on the microfacet distribution.

        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number used for branch selection.
            u2 (vec2): A 2D random sample used for sampling.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).
            sample_flags (ti.i32): Flags specifying which components to sample.

        Returns:
            BSDFSample: The sampled BSDF value.
        """
        # Choose the smooth vs. rough branch based on the microfacet distribution.
        bs = BSDFSample(vec3(0.0), vec3(0.0), 0.0, 0, 1.0)
        if self.eta == 1.0 or self.mf_distrib.effectively_smooth():
            bs = self.sample_f_Smooth(wo, uc, u2, mode, sample_flags)
        else:
            bs = self.sample_f_Rough(wo, uc, u2, mode, sample_flags)
        return bs

    @ti.func
    def alt_sample_f_Smooth(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        """
        Handles the perfect specular case for smooth dielectric materials.

        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number used for branch selection.
            u2 (vec2): A 2D random sample used for sampling.
            mode (ti.i32): Mode flag.
            sample_flags (ti.i32): Flags specifying which components to sample.

        Returns:
            BSDFSample: The sampled BSDF value.
        """
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
        """
        Samples the smooth branch of the dielectric BSDF to generate a BSDFSample, computing reflection or refraction using Fresnel equations.

        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number used for branch selection.
            u: A random sample used for the sampling procedure.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).
            sample_flags (ti.i32): Flags specifying which components to sample.

        Returns:
            BSDFSample: The sampled BSDF value.
        """
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
        """
        Samples the rough branch of the dielectric BSDF by sampling a microfacet normal from the Trowbridge-Reitz distribution and computing the Fresnel terms.

        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number used for branch selection.
            u2 (vec2): A 2D random sample used for microfacet sampling.
            mode (ti.i32): Mode flag.
            sample_flags (ti.i32): Flags specifying which components to sample.

        Returns:
            BSDFSample: The sampled BSDF value.
        """
        # Initialize output.    
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
        """
        Computes the BSDF value for a dielectric material given outgoing and incident directions.

        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).

        Returns:
            vec3: The computed BSDF value.
        """

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
        """
        Computes the probability density function (PDF) for sampling the dielectric BSDF given outgoing and incident directions.

        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag.
            sample_flags (ti.i32): Flags indicating which components are being sampled.

        Returns:
            ti.f32: The probability density of the sample.
        """
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
        """
        Returns the BxDF flags for the dielectric BSDF based on its reflection and transmission properties.

        Returns:
            ti.i32: The BxDF flag value for the dielectric material.
        """
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
    """
    Represents the BSDF for a conductor (metal), utilizing complex Fresnel calculations.
    """
    eta: vec3
    k:   vec3
    mf_distrib: TrowbridgeReitzDistribution


    @ti.func
    def sample_f(self, wo: vec3, uc: ti.f32, u2: vec2, mode: ti.i32, sample_flags: ti.i32) -> BSDFSample:
        """
        Samples the BSDF to generate a BSDFSample for conductor materials using either smooth (perfect mirror) or rough microfacet reflection.
        
        Args:
            wo (vec3): Outgoing direction.
            uc (ti.f32): A random number used for branch selection.
            u2 (vec2): A 2D random sample for microfacet normal sampling.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).
            sample_flags (ti.i32): Flags specifying which components to sample (e.g., reflection).
        
        Returns:
            BSDFSample: The sampled BSDF value including spectral value, incident direction, PDF, and flags.
        """
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
        """
        Computes the BSDF value for a conductor given outgoing and incident directions.
        
        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag (e.g., RADIANCE or IMPORTANCE).
        
        Returns:
            vec3: The computed BSDF value.
        """
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
        """
        Computes the probability density function (PDF) for the conductor BSDF given outgoing and incident directions.
        
        Args:
            wo (vec3): Outgoing direction.
            wi (vec3): Incident direction.
            mode (ti.i32): Mode flag.
            sample_flags (ti.i32): Flags indicating which components are being sampled.
        
        Returns:
            ti.f32: The probability density of the sample.
        """
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
        """
        Returns the BxDF flags for the conductor BSDF based on its microfacet roughness and reflection properties.
        
        Returns:
            ti.i32: A flag indicating whether the BSDF is specular or glossy, combined with reflection.
        """
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
    """
    Represents a complete BSDF, encapsulating multiple BxDF components (diffuse, diffuse transmission,
    dielectric, conductor) along with a local coordinate frame for transforming between world and local space.
    """
    type: ti.i32

    diffuse: DiffuseBxDF
    diffuse_transmission: DiffuseTransmissionBxDF
    dielectric: DielectricBxDF
    conductor: ConductorBxDF

    frame: Frame

    @ti.func
    def to_local(self, v: vec3) -> vec3:
        """
        Transforms a world-space vector to the local BSDF coordinate frame.
        
        Args:
            v (vec3): The world-space vector.
        
        Returns:
            vec3: The vector in the local coordinate frame.
        """
        return self.frame.to_local(v)

    @ti.func
    def from_local(self, v: vec3) -> vec3:
        """
        Transforms a vector from the local BSDF coordinate frame to world-space.
        
        Args:
            v (vec3): The local-space vector.
        
        Returns:
            vec3: The vector in world-space.
        """
        return self.frame.from_local(v)

    @ti.func
    def init_frame(self, normal):
        """
        Initializes the BSDF's local coordinate frame using the given surface normal.
        
        Args:
            normal: The surface normal used to initialize the coordinate frame.
        """
        self.frame = frame_from_z(normal)

    @ti.func
    def add_diffuse(self, R):
        """
        Sets the diffuse component of the BSDF with reflectance R and sets the BSDF type to diffuse.
        
        Args:
            R: The reflectance (vec3) for the diffuse component.
        """
        self.diffuse.R = R
        self.type = 0

    @ti.func
    def add_transmission(self, R, T):
        """
        Sets the diffuse transmission component of the BSDF with reflectance R and transmittance T,
        and sets the BSDF type to transmission.
        
        Args:
            R: The reflectance (vec3) for the transmission component.
            T: The transmittance (vec3) for the transmission component.
        """
        self.diffuse_transmission.R = R
        self.diffuse_transmission.T = T
        self.type = 1

    @ti.func
    def add_dielectric(self, eta, color, uroughness, vroughness):
        """
        Sets the dielectric component of the BSDF with the given eta, color, and roughness values,
        initializes the microfacet distribution, and sets the BSDF type to dielectric.
        
        Args:
            eta: The index of refraction.
            color: The tint (vec3).
            uroughness: The roughness value along the u direction.
            vroughness: The roughness value along the v direction.
        """
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
        """
        Sets the conductor component of the BSDF with the given eta, k, and roughness values,
        initializes the microfacet distribution, and sets the BSDF type to conductor.
        
        Args:
            eta: The refractive index (vec3) for the conductor.
            k: The extinction coefficient (vec3) for the conductor.
            uroughness: The roughness value along the u direction.
            vroughness: The roughness value along the v direction.
        """
        # print("Adding conductor", eta, k, uroughness, vroughness)
        self.conductor.eta = eta
        self.conductor.k = k
        alpha_x = self.conductor.mf_distrib.roughness_to_alpha(uroughness)
        alpha_y = self.conductor.mf_distrib.roughness_to_alpha(vroughness)
        self.conductor.mf_distrib.initialize(alpha_x, alpha_y)
        self.type = 3

    @ti.func
    def f(self, wo_world, wi_world, mode=1):
        """
        Evaluates the BSDF function for given world-space outgoing and incident directions and a specified mode.
        
        Args:
            wo_world: The outgoing direction in world-space.
            wi_world: The incident direction in world-space.
            mode (optional): The transport mode (default is 1).
        
        Returns:
            vec3: The evaluated BSDF value.
        """
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
        """
        Samples the BSDF to generate a BSDFSample, transforming the sampled direction from local to world-space.
        
        Args:
            wo_world: The outgoing direction in world-space.
            u: A random number used for sampling.
            u2: A 2D random sample used for hemisphere sampling.
            mode (optional): The transport mode (default is 1).
            sample_flags (optional): Flags specifying which components to sample (default is BXDF_ALL).
        
        Returns:
            BSDFSample: The sampled BSDF value.
        """
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
        """
        Computes the probability density function (PDF) for sampling the BSDF given world-space outgoing
        and incident directions.
        
        Args:
            wo_world: The outgoing direction in world-space.
            wi_world: The incident direction in world-space.
            mode (optional): The transport mode (default is 1).
            sample_flags (optional): Flags specifying which components are considered (default is BXDF_ALL).
        
        Returns:
            ti.f32: The computed PDF value.
        """
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
        """
        Returns the BxDF flags for the current BSDF based on the selected component type.
        
        Returns:
            ti.i32: The BxDF flag value.
        """
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