"""
This module defines the Ray class and functions for ray manipulation used in light transport simulation.
It includes methods for computing points along a ray, offsetting ray origins to avoid self-intersections,
and spawning new rays.
"""

import taichi as ti
from taichi.math import vec3, inf



@ti.dataclass
class Ray:
    """
    Represents a ray with an origin and a direction.
    Provides a method to compute a point along the ray at a given distance.
    """
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t):
        """
        Computes a point along the ray at a given distance t from the origin.

        Args:
            t (float): The distance from the ray origin.

        Returns:
            vec3: The point along the ray.
        """
        return self.origin + t * self.direction


@ti.func
def offset_ray_origin(p, n, w):
    """
    Computes an offset origin for a ray to avoid self-intersections.

    Args:
        p (vec3): The original point.
        n (vec3): The surface normal at the point p.
        w (vec3): The incident or outgoing direction used to determine the offset direction.

    Returns:
        vec3: The offset ray origin.
    """
    # Compute the error offset
    epsilon = 1e-4
    offset = n * epsilon
    if w.dot(n) < 0:
        offset = -offset

    po = p + offset

    return po


@ti.func
def spawn_ray(p, n, d):
    """
    Spawns a new ray from a point p with an offset along the normal to avoid self-intersections.

    Args:
        p (vec3): The original point.
        n (vec3): The surface normal at point p.
        d (vec3): The desired ray direction.

    Returns:
        Ray: The newly spawned ray with an offset origin.
    """
    origin = offset_ray_origin(p, n, d)
    return Ray(origin, d)
