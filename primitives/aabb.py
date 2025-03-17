"""
This module implements an Axis-Aligned Bounding Box (AABB) class and associated functions for
bounding box manipulation in light transport simulation. It includes methods for updating the
centroid, testing ray intersections, performing union operations, and computing the surface area.
"""

import taichi as ti
from taichi.math import vec3, isinf

from primitives.primitives import Primitive
from utils.constants import INF


@ti.dataclass
class AABB:
    """
    Represents an Axis-Aligned Bounding Box (AABB) used in light transport simulation.
    Provides methods for updating its centroid, testing for ray intersections, performing union operations,
    and computing the surface area.
    """
    
    min_point: vec3
    max_point: vec3
    centroid: vec3

    @ti.func
    def update_centroid(self):
        """
        Updates the centroid of the AABB based on its current min and max points.
        """
        self.centroid = (self.min_point + self.max_point) * 0.5
        # print(f"Updated centroid: {self.centroid}")

    @ti.func
    def aabb_intersect(self, ray):
        """
        Tests for an intersection between the AABB and a given ray using the slab method.
        
        Args:
            ray: The ray object with origin and direction attributes.
        
        Returns:
            bool: True if the ray intersects the AABB, otherwise False.
        """
        t_min = 0.0
        t_max = INF
        ray_inv_dir = 1 / ray.direction
        for i in range(3):
            t1 = (self.min_point[i] - ray.origin[i]) * ray_inv_dir[i]
            t2 = (self.max_point[i] - ray.origin[i]) * ray_inv_dir[i]
            t_min = ti.min(ti.max(t1, t_min), ti.max(t2, t_min))
            t_max = ti.max(ti.min(t1, t_max), ti.min(t2, t_max))
        return t_min <= t_max

    @ti.func
    def get_largest_dim(self):
        """
        Determines the index of the largest dimension of the AABB based on the extents along each axis.
        
        Returns:
            int: The index (0 for x, 1 for y, or 2 for z) of the largest dimension.
        """
        to_return = 0
        dx = abs(self.max_point[0] - self.min_point[0])
        dy = abs(self.max_point[1] - self.min_point[1])
        dz = abs(self.max_point[2] - self.min_point[2])
        if dx > dy and dx > dz:
            to_return = 0
        elif dy > dz:
            to_return = 1
        else:
            to_return = 2
        # print(f"Largest dimension: {to_return} with dx={dx}, dy={dy}, dz={dz}")
        return to_return

    @ti.func
    def offset(self, point):
        """
        Computes a normalized offset for a given point relative to the AABB's min and max points.
        
        Args:
            point (vec3): The point to be offset.
        
        Returns:
            vec3: The normalized offset vector.
        """
        o = point - self.min_point
        if self.max_point[0] > self.min_point[0]:
            o[0] /= self.max_point[0] - self.min_point[0]

        if self.max_point[1] > self.min_point[1]:
            o[1] /= self.max_point[1] - self.min_point[1]

        if self.max_point[2] > self.min_point[2]:
            o[2] /= self.max_point[2] - self.min_point[2]

        return o

    @ti.func
    def get_surface_area(self):
        """
        Computes and returns the surface area of the AABB.
        
        Returns:
            float: The computed surface area.
        """
        diagonal = self.max_point - self.min_point
        surface_area = 2 * (diagonal[0] * diagonal[1] + diagonal[0] * diagonal[2] + diagonal[1] * diagonal[2])
        # print(f"Surface area: {surface_area}")
        return surface_area

    @ti.func
    def is_empty_box(self):
        """
        Determines if the AABB is empty.
        
        Returns:
            bool: True if the AABB is empty, otherwise False.
        """
        return (self.min_point[0]==INF) and (self.max_point[0]==INF) and (self.min_point[0] > self.max_point[0])

    @ti.func
    def union_p(self, p):
        """
        Expands the AABB to include a given point and updates the centroid.
        
        Args:
            p (vec3): The point to union with the current AABB.
        
        Returns:
            AABB: The updated AABB.
        """
        self.min_point = ti.min(self.min_point, p)
        self.max_point = ti.max(self.max_point, p)
        self.update_centroid()
        # print(f"Union with point: {p}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
        return self

    @ti.func
    def union(self, b):
        """
        Expands the AABB to include another AABB and updates the centroid.
        
        Args:
            b (AABB): The other AABB to union with.
        
        Returns:
            AABB: The updated AABB.
        """
        self.min_point = ti.min(self.min_point, b.min_point)
        self.max_point = ti.max(self.max_point, b.max_point)
        self.update_centroid()
        # print(f"Union with AABB: min_point={b.min_point}, max_point={b.max_point}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
        return self

    @ti.func
    def contains(self, other):
        """
        Checks if the current AABB completely contains another AABB.
        
        Args:
            other (AABB): The AABB to check for containment.
        
        Returns:
            bool: True if the current AABB contains the other, otherwise False.
        """
        return (self.min_point[0] <= other.min_point[0] and
                self.min_point[1] <= other.min_point[1] and
                self.min_point[2] <= other.min_point[2] and
                self.max_point[0] >= other.max_point[0] and
                self.max_point[1] >= other.max_point[1] and
                self.max_point[2] >= other.max_point[2])

    @ti.func
    def equal_bounds(self):
        """
        Determines if the AABB has equal min and max bounds.
        
        Returns:
            int: 1 if bounds are equal, 0 otherwise.
        """
        equal = 1
        for i in range(3):
            if self.max_point[i] != self.min_point[i]:
                equal = 0
                break
        return equal



@ti.dataclass
class BVHPrimitive:
    """
    Represents a primitive along with its associated bounding box for use in a Bounding Volume Hierarchy (BVH).
    """
    
    prim: Primitive
    prim_num: ti.i32
    bounds: AABB


@ti.func
def union(b1, b2):
    """
    Computes and returns a new AABB representing the union of two AABBs.
    
    Args:
        b1 (AABB): The first AABB.
        b2 (AABB): The second AABB.
    
    Returns:
        AABB: The union of the two AABBs.
    """
    b3 = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
    b3.min_point = ti.min(b1.min_point, b2.min_point)
    b3.max_point = ti.max(b1.max_point, b2.max_point)
    b3.update_centroid()
    # print(f"Union with AABB: min_point={b.min_point}, max_point={b.max_point}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
    return b3


@ti.func
def union_p(b1, p1):
    """
    Computes and returns a new AABB representing the union of an AABB and a point.
    
    Args:
        b1 (AABB): The initial AABB.
        p1 (vec3): The point to include in the AABB.
    
    Returns:
        AABB: The updated AABB that includes the point.
    """
    b2 = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
    b2.min_point = ti.min(b1.min_point, p1)
    b2.max_point = ti.max(b1.max_point, p1)
    b2.update_centroid()
    # print(f"Union with AABB: min_point={b.min_point}, max_point={b.max_point}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
    return b2


@ti.func
def intersect_bounds(aabb, ray, inv_dir):
    """
    Tests whether a given ray intersects with an AABB using the inverse ray direction.
    
    Args:
        aabb (AABB): The axis-aligned bounding box to test against.
        ray: The ray with origin and direction.
        inv_dir (vec3): The precomputed inverse of the ray's direction.
    
    Returns:
        int: 1 if the ray intersects the AABB, 0 otherwise.
    """
    result = 0
    tmin = (aabb.min_point[0] - ray.origin[0]) * inv_dir[0]
    tmax = (aabb.max_point[0] - ray.origin[0]) * inv_dir[0]

    if inv_dir[0] < 0:
        tmin, tmax = tmax, tmin

    tymin = (aabb.min_point[1] - ray.origin[1]) * inv_dir[1]
    tymax = (aabb.max_point[1] - ray.origin[1]) * inv_dir[1]

    if inv_dir[1] < 0:
        tymin, tymax = tymax, tymin

    if (tmin > tymax) or (tymin > tmax):
        result = 0

    else:
        if tymin > tmin:
            tmin = tymin

        if tymax < tmax:
            tmax = tymax

        tzmin = (aabb.min_point[2] - ray.origin[2]) * inv_dir[2]
        tzmax = (aabb.max_point[2] - ray.origin[2]) * inv_dir[2]

        if inv_dir[2] < 0:
            tzmin, tzmax = tzmax, tzmin

        if (tmin > tzmax) or (tzmin > tmax):
            result = 0

        else:
            result = 1

    return result