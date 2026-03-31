"""Orbit camera controlled by hand gestures and keyboard."""
import numpy as np
import math


class OrbitCamera:
    """
    Spherical orbit camera that keeps a target point in frame.

    Coordinate system (OpenGL, Y-up):
        x = East
        y = Up (building height)
        z = South  (camera looks along -Z by default)
    """

    def __init__(self, distance: float = 400.0):
        self.target   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.azimuth  = 30.0    # degrees, horizontal rotation
        self.elevation = 25.0   # degrees — low angle like a tall person looking across
        self.distance  = distance
        self.fov       = 60.0   # vertical FOV degrees

        # Velocity for smooth inertia
        self._az_vel   = 0.0
        self._el_vel   = 0.0
        self._pan_vel  = np.zeros(2, dtype=np.float32)
        self._zoom_vel = 0.0
        self._damping  = 0.85

    # ------------------------------------------------------------------
    #  Hand-gesture-driven update (called once per frame)
    # ------------------------------------------------------------------

    def apply_gesture(self, gesture, dt: float):
        """
        gesture dict from HandTracker.get_state():
          mode       : 'idle' | 'move'
          pan_vel    : (vx, vz)  — smooth pan velocity in normalised coords
          orbit_vel  : float     — smooth horizontal orbit velocity
          zoom_vel   : float     — smooth zoom velocity
          grab       : bool      — True when fist is closed (for building grab)
          point_uv   : (u,v)|None — index fingertip screen coords
        """
        if gesture is None:
            gesture = {'mode': 'idle'}

        mode = gesture.get('mode', 'idle')

        if mode == 'idle':
            self._az_vel   *= 0.15
            self._el_vel   *= 0.15
            self._pan_vel  *= 0.15
            self._zoom_vel *= 0.15
            return

        # Direct pointing navigation (from index finger orientation)
        point_dir = gesture.get('point_dir')
        if point_dir is not None:
            forward_dir, right_dir, elev_dir = point_dir
            right = self._right_vector()
            forward = self._forward_xz()
            self.target += (forward * forward_dir + right * right_dir) * self.distance * 1.2 * dt
            self.elevation = np.clip(self.elevation + elev_dir * 45.0 * dt, 5.0, 89.0)

        # Clench-based zoom (open->far, closed->near)
        clench = gesture.get('clench')
        if clench is not None:
            min_dist = 20.0
            max_dist = 4500.0
            target_dist = np.interp(clench, [0.0, 1.0], [max_dist, min_dist])
            self.distance += (target_dist - self.distance) * min(1.0, dt * 4.0)

        # Pan from hand movement
        pvx, pvz = gesture.get('pan_vel', (0.0, 0.0))
        self._pan_vel[0] += pvx * self.distance * 1.2
        self._pan_vel[1] += pvz * self.distance * 1.2

        # Orbit from hand horizontal sweep
        orbit = gesture.get('orbit_vel', 0.0)
        self._az_vel += orbit * 120.0

        # Zoom from two-hand pinch
        zoom = gesture.get('zoom_vel', 0.0)
        self._zoom_vel += zoom * self.distance * 2.0

        # Integrate
        self.azimuth  += self._az_vel  * dt
        self.elevation = np.clip(self.elevation + self._el_vel * dt, 5.0, 89.0)
        self.distance  = np.clip(self.distance  + self._zoom_vel * dt, 20.0, 4000.0)

        right   = self._right_vector()
        forward = self._forward_xz()
        self.target += (right   * self._pan_vel[0] * dt
                     -  forward * self._pan_vel[1] * dt)

        self._az_vel   *= self._damping
        self._el_vel   *= self._damping
        self._pan_vel  *= self._damping
        self._zoom_vel *= self._damping

    # ------------------------------------------------------------------
    #  Keyboard-driven update (called once per frame)
    # ------------------------------------------------------------------

    def apply_keyboard(self, keys_held: set, dt: float):
        """WASD = pan, Arrow keys = orbit, scroll/+- = zoom."""
        pan_speed   = self.distance * 0.8
        orbit_speed = 90.0

        right   = self._right_vector()
        forward = self._forward_xz()
        if 'W' in keys_held:
            self.target -= forward * pan_speed * dt
        if 'S' in keys_held:
            self.target += forward * pan_speed * dt
        if 'A' in keys_held:
            self.target -= right * pan_speed * dt
        if 'D' in keys_held:
            self.target += right * pan_speed * dt

        if 'LEFT' in keys_held:
            self.azimuth -= orbit_speed * dt
        if 'RIGHT' in keys_held:
            self.azimuth += orbit_speed * dt
        if 'UP' in keys_held:
            self.elevation = min(self.elevation + orbit_speed * 0.6 * dt, 89.0)
        if 'DOWN' in keys_held:
            self.elevation = max(self.elevation - orbit_speed * 0.6 * dt, 5.0)

    # ------------------------------------------------------------------
    #  Matrix helpers
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        x = math.cos(el) * math.sin(az)
        y = math.sin(el)
        z = math.cos(el) * math.cos(az)
        return self.target + np.array([x, y, z], dtype=np.float32) * self.distance

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.target, np.array([0, 1, 0], dtype=np.float32))

    def proj_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(math.radians(self.fov), aspect, 1.0, 8000.0)

    def unproject_ray(self, u: float, v: float, aspect: float):
        """Return (origin, direction) world-space ray for normalised screen coord (u,v)."""
        proj = self.proj_matrix(aspect)
        view = self.view_matrix()
        inv_vp = np.linalg.inv(proj @ view)
        ndc = np.array([u * 2 - 1, 1 - v * 2, -1.0, 1.0], dtype=np.float64)
        world = inv_vp @ ndc
        world /= world[3]
        direction = world[:3] - self.position
        direction /= np.linalg.norm(direction)
        return self.position.copy(), direction

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _right_vector(self):
        az = math.radians(self.azimuth)
        return np.array([math.cos(az), 0.0, -math.sin(az)], dtype=np.float32)

    def _forward_xz(self):
        az = math.radians(self.azimuth)
        return np.array([math.sin(az), 0.0, math.cos(az)], dtype=np.float32)


# ------------------------------------------------------------------
#  Pure-numpy math (no GLM dependency)
# ------------------------------------------------------------------

def _look_at(eye, center, up):
    f = center - eye;   f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r;  m[0, 3] = -r.dot(eye)
    m[1, :3] = u;  m[1, 3] = -u.dot(eye)
    m[2, :3] = -f; m[2, 3] =  f.dot(eye)
    return m


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m
