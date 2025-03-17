"""
This module implements various sampling functions and data structures for light transport simulation.
It provides uniform sampling functions for spheres, hemispheres, disks (both concentric and polar),
cosine-weighted hemisphere sampling, and utilities to compute probability density functions (PDFs).
"""

import taichi as ti
from taichi.math import pi, length, normalize, dot, vec3, sqrt, vec2, sin, cos

from utils.constants import inv_pi


@ti.dataclass
class ShapeSample:
    p: vec3  # Point of intersection
    n: vec3  # Normal at intersection
    pdf: ti.f32  # Probability density of the sample

    """
    Represents a sample on a shape, containing the intersection point, normal, and the PDF value.
    """


@ti.func
def sample_uniform_sphere(u):
    """
    Uniformly samples a point on the surface of a sphere given a 2D uniform random sample.

    Args:
        u (vec2): A 2D vector with components in [0, 1] used for sampling.

    Returns:
        vec3: A point on the sphere's surface.
    """
    z = 1 - 2 * u[0]
    r = sqrt(max(0.0, 1 - z * z))
    phi = 2 * pi * u[1]
    return vec3(r * cos(phi), r * sin(phi), z)


@ti.func
def uniform_sphere_pdf():
    """
    Returns the probability density function (PDF) for uniform sampling on a sphere.

    Returns:
        float: The PDF value for a uniform sphere.
    """
    return 1 / (4 * pi)


@ti.func
def sample_uniform_hemisphere(u2):
    """
    Uniformly samples a point on the surface of a hemisphere given a 2D uniform random sample.

    Args:
        u2 (vec2): A 2D vector with components in [0, 1] used for sampling.

    Returns:
        vec3: A point on the hemisphere's surface.
    """
    z = u2[0]
    r = sqrt(max(0.0, 1.0 - z * z))
    phi = 2 * pi * u2[1]
    x = r * cos(phi)
    y = r * sin(phi)
    return vec3(x, y, z)


@ti.func
def uniform_hemisphere_pdf():
    """
    Returns the probability density function (PDF) for uniform sampling on a hemisphere.

    Returns:
        float: The PDF value for a uniform hemisphere.
    """
    return 1.0 / (2.0 * pi)


@ti.func
def sample_uniform_disk_concentric(u):
    """
    Samples a point on a unit disk using a concentric mapping, useful for lens simulation.

    Args:
        u (vec2): A 2D vector with components in [0, 1] used for sampling.

    Returns:
        vec2: A point on the unit disk in concentric coordinates.
    """
    r = sqrt(u[0])
    theta = 2 * pi * u[1]
    return vec2(r * cos(theta), r * sin(theta))


@ti.func
def get_shape_pdf(self, intr, wi):
    """
    Computes the PDF for a shape sample and converts it to a solid angle measure.

    Args:
        self: The object containing a 'center' attribute (assumed to be part of a shape).
        intr: The intersection data containing the intersected point and primitive information.
        wi (vec3): The incident light direction.

    Returns:
        float: The PDF value converted to a solid angle, or 0.0 if the cosine term is non-positive.
    """
    pdf = intr.primitive.get_pdf()
    # convert to solid angle
    to_center = self.center[None] - intr.intersected_point
    distance = length(to_center)
    cos_theta = dot(normalize(to_center), wi)
    if cos_theta > 0:
        return ((distance * distance) / cos_theta) * pdf
    return 0.0


@ti.func
def sample_uniform_disk_polar(u):
    """
    Uniformly samples a point on a disk using polar coordinates.

    Args:
        u (vec2): A 2D vector with components in [0, 1] used for sampling.

    Returns:
        vec2: A point on the disk sampled in polar coordinates.
    """
    r = ti.sqrt(u[0])
    theta = 2 * pi * u[1]
    return vec2(r * cos(theta), r * sin(theta))


@ti.func
def concentric_sample_disk(u):
    """
    Samples a point on a disk using a concentric mapping approach to reduce clustering.

    Args:
        u (vec2): A 2D vector with components in [0, 1] used for sampling.

    Returns:
        vec2: A point on the disk using the concentric mapping technique.
    """
    u_offset = 2.0 * u - vec2(1, 1)
    to_return = vec2(0, 0)
    if u_offset.x != 0 and u_offset.y != 0:
        r, theta = 0.0, 0.0
        if ti.abs(u_offset.x) > ti.abs(u_offset.y):
            r = u_offset.x
            theta = pi / 4 * (u_offset.y / u_offset.x)
        else:
            r = u_offset.y
            theta = pi / 2 - pi / 4 * (u_offset.x / u_offset.y)
        to_return = r * vec2(cos(theta), sin(theta))
    return to_return


@ti.func
def sample_cosine_hemisphere(u):
    """
    Samples a point on a hemisphere using cosine-weighted distribution based on a concentric disk sample.

    Args:
        u (vec2): A 2D vector with components in [0, 1] used for sampling.

    Returns:
        vec3: A point on the hemisphere with cosine weighting.
    """
    d = concentric_sample_disk(u)
    z = ti.sqrt(ti.max(0.0, 1 - d.x * d.x - d.y * d.y))
    return vec3(d.x, d.y, z)


@ti.func
def cosine_hemisphere_pdf(cos_theta):
    """
    Computes the probability density function (PDF) for cosine-weighted hemisphere sampling.

    Args:
        cos_theta (float): The cosine of the angle between the sampled direction and the surface normal.

    Returns:
        float: The PDF value for the cosine-weighted hemisphere.
    """
    return cos_theta * inv_pi