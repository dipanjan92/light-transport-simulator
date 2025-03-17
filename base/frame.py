"""
This module implements a coordinate frame system for light transport simulation.
It provides the Frame class for representing an orthonormal basis and various helper functions
to construct frames from different input vectors, as well as functions for vector coordinate
transformations.
"""

import taichi as ti
from taichi.math import vec3, dot, cross, normalize, length


@ti.dataclass
class Frame:
    """
    Represents an orthonormal coordinate frame defined by three basis vectors: x, y, and z.
    Provides methods to transform vectors between global and local coordinate systems.
    """
    x: ti.types.vector(3, ti.f32)
    y: ti.types.vector(3, ti.f32)
    z: ti.types.vector(3, ti.f32)

    @ti.func
    def to_local(self, v):
        """
        Transforms a vector from the global coordinate system to the local frame coordinates.
        
        Args:
            v (vec3): The vector in global coordinates.
        
        Returns:
            vec3: The vector expressed in the local frame.
        """
        return vec3(dot(v, self.x), dot(v, self.y), dot(v, self.z))

    @ti.func
    def from_local(self, v):
        """
        Transforms a vector from the local frame coordinates to the global coordinate system.
        
        Args:
            v (vec3): The vector in local frame coordinates.
        
        Returns:
            vec3: The vector expressed in global coordinates.
        """
        return self.x * v[0] + self.y * v[1] + self.z * v[2]


@ti.func
def create_frame(x, y, z):
    """
    Creates a new Frame given three normalized and mutually orthogonal basis vectors x, y, and z.
    Validates that the vectors are normalized and orthogonal.
    
    Args:
        x (vec3): The first basis vector.
        y (vec3): The second basis vector.
        z (vec3): The third basis vector.
    
    Returns:
        Frame: The constructed coordinate frame.
    """
    assert ti.abs(length(x) - 1.0) < 1e-4, f"x is not normalized: {length(x)}"
    assert ti.abs(length(y) - 1.0) < 1e-4, f"y is not normalized: {length(y)}"
    assert ti.abs(length(z) - 1.0) < 1e-4, f"z is not normalized: {length(z)}"
    assert ti.abs(dot(x, y)) < 1e-4, "x and y are not orthogonal"
    assert ti.abs(dot(y, z)) < 1e-4, "y and z are not orthogonal"
    assert ti.abs(dot(z, x)) < 1e-4, "z and x are not orthogonal"
    return Frame(x=x, y=y, z=z)


@ti.func
def copysign(x, y):
    """
    Returns the absolute value of x with the sign of y.
    
    Args:
        x (float): The value whose magnitude is used.
        y (float): The value whose sign is applied.
    
    Returns:
        float: The value of |x| with the sign of y.
    """
    return ti.abs(x) if y >= 0 else -ti.abs(x)


@ti.func
def coordinate_system(v):
    """
    Generates an orthonormal coordinate system given a vector v.
    Returns two vectors that, together with v, form an orthonormal basis.
    
    Args:
        v (vec3): The input vector to base the coordinate system on.
    
    Returns:
        (vec3, vec3): Two vectors that form an orthonormal basis with v.
    """
    sign = copysign(1.0, v.z)
    a = -1.0 / (sign + v.z)
    b = v.x * v.y * a
    v2 = vec3(1.0 + sign * v.x * v.x * a, sign * b, -sign * v.x)
    v3 = vec3(b, sign + v.y * v.y * a, -v.y)
    return v2, v3


@ti.func
def frame_from_xz(x, z):
    """
    Constructs a Frame using the provided x and z vectors. The x vector is re-orthogonalized relative to z.
    
    Args:
        x (vec3): The desired x-axis vector.
        z (vec3): The desired z-axis vector.
    
    Returns:
        Frame: The constructed coordinate frame.
    """
    x = normalize(x)
    z = normalize(z)
    x = normalize(x - dot(x, z) * z)  # Re-orthogonalize x relative to z
    y = cross(z, x)
    return create_frame(x, y, z)


@ti.func
def frame_from_xy(x, y):
    """
    Constructs a Frame using the provided x and y vectors. The z vector is computed as the cross product of x and y.
    
    Args:
        x (vec3): The desired x-axis vector.
        y (vec3): The desired y-axis vector.
    
    Returns:
        Frame: The constructed coordinate frame.
    """
    x = normalize(x)
    y = normalize(y)
    z = cross(x, y)
    return create_frame(x, y, z)


@ti.func
def frame_from_z(z):
    """
    Constructs a Frame from the given z vector by generating an orthonormal basis.
    
    Args:
        z (vec3): The desired z-axis vector.
    
    Returns:
        Frame: The constructed coordinate frame.
    """
    z = normalize(z)
    x, y = coordinate_system(z)
    return create_frame(x, y, z)


@ti.func
def frame_from_x(x):
    """
    Constructs a Frame from the given x vector by generating an orthonormal basis.
    
    Args:
        x (vec3): The desired x-axis vector.
    
    Returns:
        Frame: The constructed coordinate frame.
    """
    x = normalize(x)
    y, z = coordinate_system(x)
    return create_frame(x, y, z)


@ti.func
def frame_from_y(y):
    """
    Constructs a Frame from the given y vector by generating an orthonormal basis.
    
    Args:
        y (vec3): The desired y-axis vector.
    
    Returns:
        Frame: The constructed coordinate frame.
    """
    y = normalize(y)
    z, x = coordinate_system(y)
    return create_frame(x, y, z)
