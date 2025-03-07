import taichi as ti
from taichi.math import vec3, vec4, normalize, dot, cross, radians, tan, length, pi
from base.frame import Frame, create_frame
from primitives.ray import Ray


@ti.func
def sample_uniform_disk_concentric(u, v):
    """Samples a point on a unit disk using concentric mapping."""
    u = 2 * u - 1
    v = 2 * v - 1
    p_disk = vec3(0.0, 0.0, 0.0)

    if not (u == 0 and v == 0):
        theta = 0.0
        r = 0.0
        if ti.abs(u) > ti.abs(v):
            r = u
            theta = (pi / 4) * (v / u)
        else:
            r = v
            theta = (pi / 2) - (pi / 4) * (u / v)
        p_disk = vec3(r * ti.cos(theta), r * ti.sin(theta), 0.0)

    return p_disk


@ti.dataclass
class PerspectiveCamera:
    """PBRT-style Perspective Camera"""
    width: ti.i32
    height: ti.i32
    position: vec3
    frame: Frame
    fov: ti.f32
    aspect_ratio: ti.f32
    lens_radius: ti.f32
    focal_distance: ti.f32
    screen_window: vec4  # (x_min, x_max, y_min, y_max)
    dx_camera: vec3
    dy_camera: vec3

    @ti.func
    def camera_from_raster(self, p_film):
        """Converts raster space coordinates to camera space."""
        x_min, x_max, y_min, y_max = self.screen_window.x, self.screen_window.y, self.screen_window.z, self.screen_window.w

        # Convert raster space to screen space
        x = (p_film.x / self.width) * (x_max - x_min) + x_min
        y = (p_film.y / self.height) * (y_max - y_min) + y_min

        return vec3(x, y, -1.0)  # ✅ Ensure -Z film plane (PBRT convention)

    @ti.func
    def generate_ray(self, s: ti.f32, t: ti.f32):
        """Generates a camera ray in world space."""
        p_film = vec3(s * self.width, t * self.height, 0.0)
        p_camera = self.camera_from_raster(p_film)

        ray_origin = vec3(0.0, 0.0, 0.0)
        ray_dir = normalize(p_camera - ray_origin)  # ✅ Ensure correct ray direction

        if self.lens_radius > 0:
            lens_u, lens_v = ti.random(), ti.random()
            p_lens = self.lens_radius * sample_uniform_disk_concentric(lens_u, lens_v)
            ft = self.focal_distance / ti.abs(ray_dir.z)
            p_focus = ray_origin + ft * ray_dir
            ray_origin = vec3(p_lens.x, p_lens.y, 0.0)
            ray_dir = normalize(p_focus - ray_origin)

        world_ray_origin = self.position + self.frame.from_local(ray_origin)
        world_ray_dir = normalize(self.frame.from_local(ray_dir))

        return world_ray_origin, world_ray_dir

    @ti.func
    def generate_ray_differential(self, s: ti.f32, t: ti.f32):
        """Generates primary ray differentials."""
        p_film = vec3(s * self.width, t * self.height, 0.0)
        p_camera = self.camera_from_raster(p_film)

        ray_origin = vec3(0.0, 0.0, 0.0)
        ray_dir = normalize(p_camera - ray_origin)

        # Compute differentials
        rx_origin, ry_origin = ray_origin, ray_origin
        rx_direction = normalize(p_camera + self.dx_camera - ray_origin)
        ry_direction = normalize(p_camera + self.dy_camera - ray_origin)

        # Apply lens-based differentials (if lens_radius > 0)
        if self.lens_radius > 0:
            lens_u, lens_v = ti.random(), ti.random()
            p_lens = self.lens_radius * sample_uniform_disk_concentric(lens_u, lens_v)
            ft = self.focal_distance / ti.abs(ray_dir.z)
            p_focus = ray_origin + ft * ray_dir
            ray_origin = vec3(p_lens.x, p_lens.y, 0.0)
            ray_dir = normalize(p_focus - ray_origin)

            dx = normalize(p_camera + self.dx_camera - ray_origin)
            ft_x = self.focal_distance / ti.abs(dx.z)
            p_focus_x = ray_origin + ft_x * dx
            rx_origin = vec3(p_lens.x, p_lens.y, 0.0)
            rx_direction = normalize(p_focus_x - rx_origin)

            dy = normalize(p_camera + self.dy_camera - ray_origin)
            ft_y = self.focal_distance / ti.abs(dy.z)
            p_focus_y = ray_origin + ft_y * dy
            ry_origin = vec3(p_lens.x, p_lens.y, 0.0)
            ry_direction = normalize(p_focus_y - ry_origin)

        return self.position + self.frame.from_local(ray_origin), normalize(self.frame.from_local(ray_dir)), \
               self.position + self.frame.from_local(rx_origin), normalize(self.frame.from_local(rx_direction)), \
               self.position + self.frame.from_local(ry_origin), normalize(self.frame.from_local(ry_direction))


@ti.func
def frame_look_at(eye, center, up):
    """Creates a PBRT-compatible camera-to-world transformation frame."""
    print(eye, center, up)
    forward = normalize(center - eye)
    if length(forward) < 1e-8:
        forward = vec3(0, 0, -1)

    right = normalize(cross(up, forward))
    up_corrected = cross(forward, right)  # Ensure orthonormality

    return create_frame(right, up_corrected, -forward)  # ✅ Flip Z-axis to match PBRT


@ti.kernel
def auto_fit_camera(scene_min: vec3, scene_max: vec3, fov_degrees: ti.f32,
                    film_width: ti.i32, film_height: ti.i32, margin: ti.f32) -> PerspectiveCamera:
    """Automatically places the camera based on PBRT world transformation."""
    center = 0.5 * (scene_min + scene_max)
    extent = (scene_max - scene_min) * (1 + margin)
    scene_radius = 0.5 * ti.max(extent.x, ti.max(extent.y, extent.z))

    print("Scene Bounding Box:", scene_min, scene_max, "Center:", center)

    half_fov = 0.5 * radians(fov_degrees)
    distance = (scene_radius / tan(half_fov)) + scene_radius

    # PBRT-style camera placement (universal for all PBRT v4 files)
    eye = center - distance * vec3(0, 0, 1)

    # Compute camera-to-world transformation
    cam_frame = frame_look_at(eye, center, vec3(0, 1, 0))

    aspect = film_width / film_height
    screen_height = 2.0 * tan(half_fov)
    screen_width = aspect * screen_height

    dx_camera = vec3(screen_width / film_width, 0, 0)
    dy_camera = vec3(0, screen_height / film_height, 0)

    # ✅ Compute screen window dynamically from FOV
    screen_window = vec4(-screen_width * 0.5, screen_width * 0.5,
                         -screen_height * 0.5, screen_height * 0.5)

    cam = PerspectiveCamera(
        width=film_width, height=film_height, position=eye, frame=cam_frame,
        fov=fov_degrees, aspect_ratio=aspect, lens_radius=0.0, focal_distance=distance,
        screen_window=screen_window, dx_camera=dx_camera, dy_camera=dy_camera
    )
    return cam

@ti.kernel
def get_camera(eye: vec3, center: vec3, up: vec3, fov_degrees: ti.f32, film_width: ti.i32, film_height: ti.i32) -> PerspectiveCamera:
    """Places the camera based on PBRT transformation matrix from `scene_dict`."""

    print("Extracted Camera Parameters:", eye, center, up, fov_degrees)

    # Compute the camera-to-world transformation frame
    cam_frame = frame_look_at(eye, center, up)

    half_fov = 0.5 * radians(fov_degrees)
    distance = length(eye - center)  # Compute focal distance from eye to center

    aspect = film_width / film_height
    screen_height = 2.0 * tan(half_fov)
    screen_width = aspect * screen_height

    dx_camera = vec3(screen_width / film_width, 0, 0)
    dy_camera = vec3(0, screen_height / film_height, 0)

    # Compute screen window dynamically from FOV
    screen_window = vec4(-screen_width * 0.5, screen_width * 0.5,
                         -screen_height * 0.5, screen_height * 0.5)

    cam = PerspectiveCamera(
        width=film_width, height=film_height, position=eye, frame=cam_frame,
        fov=fov_degrees, aspect_ratio=aspect, lens_radius=0.0, focal_distance=distance,
        screen_window=screen_window, dx_camera=dx_camera, dy_camera=dy_camera
    )
    return cam