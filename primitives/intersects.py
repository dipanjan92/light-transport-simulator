import taichi as ti
from taichi.math import vec3, normalize, dot
from primitives.primitives import Primitive
from base.frame import Frame, frame_from_z, create_frame, coordinate_system  # or whichever you prefer.

@ti.dataclass
class Intersection:
    min_distance: ti.f32
    intersected_point: vec3
    normal: vec3
    shading_normal: vec3
    # We remove dpdu/dpdv/dndu/dndv if not needed
    nearest_object: Primitive
    intersected: ti.i32

    # Store the local frame so we don't need partial derivatives
    frame: Frame

    @ti.func
    def set_intersection(self, ray, prim, t_hit):
        # 1) Basic intersection data
        self.intersected = 1
        self.min_distance = t_hit
        self.intersected_point = ray.origin + t_hit * ray.direction
        self.nearest_object = prim

        # 2) Compute geometric normal
        if prim.shape_type == 0:
            # Triangles
            self.normal = prim.triangle.normal
        elif prim.shape_type == 1:
            # Sphere
            center = prim.sphere.center
            radius = prim.sphere.radius
            self.normal = normalize((self.intersected_point - center) / radius)

        # (optional) Flip if normal faces away from incoming ray
        # out_to_in = self.normal.dot(ray.direction) < 0
        # self.normal = self.normal if out_to_in else -self.normal

        # For flat shading, shading_normal = normal
        self.shading_normal = self.normal

        self.frame = frame_from_z(normalize(self.shading_normal))
        # z_axis = normalize(self.shading_normal)
        # t, b = coordinate_system(z_axis)
        # self.frame = create_frame(t, b, z_axis)

    @ti.func
    def Le(self, d: vec3):
        """
        Emission if the object is a light and the direction is front-facing.
        """
        L = vec3(0.0)
        if self.nearest_object.shape_type == 0 and self.nearest_object.is_light:
            # e.g. area light on triangle
            if dot(self.normal, d) >= 0.0:
                L += self.nearest_object.material.emission
        return L

    @ti.func
    def get_bsdf(self):
        bsdf = self.nearest_object.bsdf
        bsdf.frame = self.frame

        return bsdf
