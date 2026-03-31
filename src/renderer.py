"""
ModernGL renderer.
All GLSL shaders are embedded as strings — no external shader files needed.
"""
import numpy as np
import moderngl
import cv2
from PIL import Image, ImageDraw, ImageFont

from .gis_loader import UrbanScene


# ------------------------------------------------------------------
#  GLSL — Background (webcam quad)
# ------------------------------------------------------------------
_BG_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_BG_FRAG = """
#version 330
uniform sampler2D u_tex;
in  vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""

# ------------------------------------------------------------------
#  GLSL — Ground plane
# ------------------------------------------------------------------
_GROUND_VERT = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos;
out vec2 v_xz;
void main() {
    gl_Position = u_vp * vec4(in_pos, 1.0);
    v_xz = in_pos.xz;
}
"""

_GROUND_FRAG = """
#version 330
uniform sampler2D u_map_tex;
uniform vec4  u_map_bounds;   // (min_x, min_z, max_x, max_z)
uniform float u_has_map;      // 1.0 = map texture available

in  vec2 v_xz;
out vec4 fragColor;

float grid(vec2 xz, float step) {
    vec2 g = abs(fract(xz / step - 0.5) - 0.5) / fwidth(xz / step);
    return 1.0 - min(min(g.x, g.y), 1.0);
}

void main() {
    // Procedural fallback
    vec3 asphalt = vec3(0.32, 0.33, 0.30);
    float kerb = grid(v_xz, 100.0) * 0.35;
    float slab = grid(v_xz, 2.0) * 0.03;
    float line = clamp(kerb + slab, 0.0, 1.0);
    vec3 pavement = vec3(0.48, 0.47, 0.43);
    vec3 proc = mix(asphalt, pavement, line);

    vec3 col = proc;

    if (u_has_map > 0.5) {
        // Map tile UV from world position
        vec2 uv = (v_xz - u_map_bounds.xy) / (u_map_bounds.zw - u_map_bounds.xy);
        // Inside map bounds?
        float inside = step(0.0, uv.x) * step(0.0, uv.y)
                      * (1.0 - step(1.0, uv.x)) * (1.0 - step(1.0, uv.y));

        if (inside > 0.5) {
            vec3 mapCol = texture(u_map_tex, uv).rgb;
            // Slight brightness boost (OSM tiles can be muted)
            mapCol = pow(mapCol, vec3(0.92)) * 1.05;
            col = mapCol;
        } else {
            // Soft blend at edges of the map
            float edgeDist = max(
                max(-uv.x, uv.x - 1.0),
                max(-uv.y, uv.y - 1.0)
            );
            float edgeFade = smoothstep(0.0, 0.08, edgeDist);
            vec2 clampedUV = clamp(uv, vec2(0.001), vec2(0.999));
            vec3 edgeMap = texture(u_map_tex, clampedUV).rgb;
            edgeMap = pow(edgeMap, vec3(0.92)) * 1.05;
            col = mix(edgeMap, proc, edgeFade);
        }
    }

    // Distance fade to haze
    float dist = length(v_xz);
    float fade = exp(-dist * 0.0003);
    col = mix(vec3(0.62, 0.68, 0.78), col, fade);

    fragColor = vec4(col, 1.0);
}
"""

# ------------------------------------------------------------------
#  GLSL — Buildings & Roads (shared Blinn-Phong)
# ------------------------------------------------------------------
_MESH_VERT = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_pos;
out vec3 v_normal;
out vec3 v_color;
void main() {
    gl_Position = u_vp * vec4(in_pos, 1.0);
    v_pos    = in_pos;
    v_normal = in_normal;
    v_color  = in_color;
}
"""

_MESH_FRAG = """
#version 330
uniform vec3  u_cam_pos;
uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;
uniform float u_ambient;
uniform float u_selected;
uniform float u_has_windows;  // 1=buildings, 0=roads/ground

in vec3 v_pos;
in vec3 v_normal;
in vec3 v_color;
out vec4 fragColor;

// Procedural window grid on facade surfaces
float windowMask(vec3 pos, vec3 N) {
    if (abs(N.y) > 0.4) return 0.0;        // skip roofs / roads
    // Facade tangent perpendicular to normal in XZ
    vec2 n2  = normalize(N.xz);
    vec2 tan = vec2(-n2.y, n2.x);
    float horiz = dot(pos.xz, tan);
    float vert  = pos.y;
    const float FLOOR_H  = 3.5;
    const float WIN_W    = 1.8;
    float fh = fract(horiz / WIN_W);
    float fv = fract(vert  / FLOOR_H);
    // Window occupies 25-82% of each cell in both axes
    float wh = step(0.25, fh) * (1.0 - step(0.82, fh));
    float wv = step(0.28, fv) * (1.0 - step(0.84, fv));
    return wh * wv;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_sun_dir);
    vec3 V = normalize(u_cam_pos - v_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.35;

    vec3 base = v_color;

    // --- Window grid ---
    float win = windowMask(v_pos, N) * u_has_windows;
    // Day = reflective blue-grey glass; night = warm lit interiors
    float dayFactor   = smoothstep(0.15, 0.40, u_ambient);
    vec3  glassDay    = vec3(0.52, 0.66, 0.84) * 1.25;
    vec3  glassNight  = vec3(1.00, 0.88, 0.52) * 1.60;
    vec3  glassColor  = mix(glassNight, glassDay, dayFactor);
    base = mix(base, glassColor, win * 0.78);

    // --- Fake AO: lower floors slightly darker ---
    float ao = 0.78 + 0.22 * smoothstep(0.0, 10.0, v_pos.y);

    // --- Edge darkening: darken faces angled away from camera for definition ---
    float edgeFactor = 1.0 - pow(1.0 - max(dot(N, V), 0.0), 2.0) * 0.25;

    // --- Selection highlight ---
    base = mix(base, vec3(1.0, 0.85, 0.15), u_selected * 0.55);

    // --- Lighting (brighter ambient, better fill) ---
    float fillLight = max(dot(N, normalize(vec3(-0.3, 0.5, -0.6))), 0.0) * 0.15;
    vec3 color = base * ao * edgeFactor *
                 (u_ambient + (1.0 - u_ambient) * (diff + fillLight) * u_sun_color)
               + spec * u_sun_color;

    // --- Atmospheric haze (distance + altitude) — reduced for clarity ---
    float camDist = length(v_pos - u_cam_pos);
    float haze    = exp(-camDist * 0.00025);
    vec3  hazeCol = mix(vec3(0.65, 0.72, 0.84), vec3(0.55, 0.65, 0.80),
                        smoothstep(0.15, 0.50, u_ambient));
    color = mix(hazeCol, color, haze);

    fragColor = vec4(color, 1.0);
}
"""

# ------------------------------------------------------------------
#  GLSL — Sky gradient (rendered when no webcam is available)
# ------------------------------------------------------------------
_SKY_VERT = """
#version 330
in vec2 in_pos;
out float v_t;
void main() {
    gl_Position = vec4(in_pos, 0.9999, 1.0);
    v_t = in_pos.y * 0.5 + 0.5;
}
"""

_SKY_FRAG = """
#version 330
uniform float u_ambient;   // proxy for time-of-day
in  float v_t;
out vec4  fragColor;
void main() {
    // Daytime: blue sky.  Low sun: orange horizon.
    float day = smoothstep(0.10, 0.45, u_ambient);
    vec3 zenithDay    = vec3(0.22, 0.44, 0.82);
    vec3 horizonDay   = vec3(0.68, 0.80, 0.92);
    vec3 zenithDusk   = vec3(0.10, 0.10, 0.28);
    vec3 horizonDusk  = vec3(0.85, 0.45, 0.15);
    vec3 zenith  = mix(zenithDusk,  zenithDay,  day);
    vec3 horizon = mix(horizonDusk, horizonDay, day);
    // Exponential gradient — denser near horizon
    float t = pow(clamp(v_t, 0.0, 1.0), 0.6);
    fragColor = vec4(mix(horizon, zenith, t), 1.0);
}
"""

# ------------------------------------------------------------------
#  GLSL — Gesture overlay HUD quad
# ------------------------------------------------------------------
_HUD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
"""

_HUD_FRAG = """
#version 330
uniform sampler2D u_tex;
in  vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = texture(u_tex, v_uv); }
"""


class Renderer:
    def __init__(self, ctx: moderngl.Context, win_size: tuple[int, int]):
        self.ctx      = ctx
        self.win_size = win_size

        # Compile programs
        self.bg_prog     = ctx.program(vertex_shader=_BG_VERT,     fragment_shader=_BG_FRAG)
        self.ground_prog = ctx.program(vertex_shader=_GROUND_VERT, fragment_shader=_GROUND_FRAG)
        self.mesh_prog   = ctx.program(vertex_shader=_MESH_VERT,   fragment_shader=_MESH_FRAG)
        self.sky_prog    = ctx.program(vertex_shader=_SKY_VERT,    fragment_shader=_SKY_FRAG)
        self.hud_prog    = ctx.program(vertex_shader=_HUD_VERT,    fragment_shader=_HUD_FRAG)

        # Sun defaults (can be animated later)
        self.sun_dir    = np.array([0.6, 1.0, 0.4], np.float32)
        self.sun_dir   /= np.linalg.norm(self.sun_dir)
        self.sun_color  = np.array([1.0, 0.96, 0.88], np.float32)
        self.ambient    = 0.40

        self._setup_bg()
        self._setup_sky()
        self._setup_ground()
        self._setup_hud()
        self._setup_webcam_texture()

        # Scene buffers (populated by upload_scene)
        self.building_vao: moderngl.VertexArray | None = None
        self.road_vao:     moderngl.VertexArray | None = None
        self.area_vao:     moderngl.VertexArray | None = None
        self.map_texture:  moderngl.Texture | None = None
        self.map_bounds    = None  # (min_x, min_z, max_x, max_z)
        self.building_aabbs = []
        self.selected_idx   = -1
        self.selected_name  = ""

        # Building grab-and-move state
        self._grabbed_idx   = -1
        self._grab_active   = False
        self._scene_verts: np.ndarray | None = None  # writable copy of building verts
        self._scene_idx:   np.ndarray | None = None
        self._vert_ranges: list = []  # [(start_vert, end_vert)] per building

    # ------------------------------------------------------------------
    #  Scene upload
    # ------------------------------------------------------------------

    def upload_scene(self, scene: UrbanScene):
        self.building_aabbs = list(scene.building_aabbs)  # mutable copy

        if len(scene.building_verts) > 0:
            self._scene_verts = scene.building_verts.copy()  # writable CPU copy
            self._scene_idx   = scene.building_idx.copy()
            self._building_vbo = self.ctx.buffer(self._scene_verts.tobytes())
            ibo = self.ctx.buffer(self._scene_idx.tobytes())
            self.building_vao = self.ctx.vertex_array(
                self.mesh_prog,
                [(self._building_vbo, '3f 3f 3f', 'in_pos', 'in_normal', 'in_color')],
                ibo,
            )
            # Build per-building vertex ranges from index buffer
            self._vert_ranges = _compute_vert_ranges(
                self._scene_idx, self._scene_verts, self.building_aabbs
            )

        if len(scene.road_verts) > 0:
            vbo = self.ctx.buffer(scene.road_verts.tobytes())
            ibo = self.ctx.buffer(scene.road_idx.tobytes())
            self.road_vao = self.ctx.vertex_array(
                self.mesh_prog,
                [(vbo, '3f 3f 3f', 'in_pos', 'in_normal', 'in_color')],
                ibo,
            )

        if len(scene.area_verts) > 0:
            vbo = self.ctx.buffer(scene.area_verts.tobytes())
            ibo = self.ctx.buffer(scene.area_idx.tobytes())
            self.area_vao = self.ctx.vertex_array(
                self.mesh_prog,
                [(vbo, '3f 3f 3f', 'in_pos', 'in_normal', 'in_color')],
                ibo,
            )

        # Map tile ground texture
        if scene.map_image is not None and scene.map_bounds is not None:
            h, w = scene.map_image.shape[:2]
            self.map_texture = self.ctx.texture((w, h), 3, scene.map_image.tobytes())
            self.map_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.map_texture.build_mipmaps()
            self.map_texture.anisotropy = 8.0
            b = scene.map_bounds
            self.map_bounds = (b['min_x'], b['min_z'], b['max_x'], b['max_z'])
            print(f"  Map texture uploaded: {w}x{h}")

    # ------------------------------------------------------------------
    #  Per-frame draw
    # ------------------------------------------------------------------

    def draw(self, cam_frame, camera, gesture: dict):
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = 'back'

        w, h = self.win_size
        vp   = (camera.proj_matrix(w / h) @ camera.view_matrix()).T.astype(np.float32)

        # ---- 1. Background: webcam feed or procedural sky ---------
        if cam_frame is not None:
            self._draw_background(cam_frame, gesture)
        else:
            ctx.clear(0.0, 0.0, 0.0)
            self._draw_sky()

        # ---- 2. Ground --------------------------------------------
        self.ground_prog['u_vp'].write(vp.tobytes())
        if self.map_texture and self.map_bounds:
            self.map_texture.use(location=2)
            self.ground_prog['u_map_tex'].value = 2
            self.ground_prog['u_map_bounds'].value = self.map_bounds
            self.ground_prog['u_has_map'].value = 1.0
        else:
            self.ground_prog['u_has_map'].value = 0.0
        ctx.disable(moderngl.CULL_FACE)
        self.ground_vao.render()
        ctx.enable(moderngl.CULL_FACE)

        # ---- 2b. Area polygons (parks, water, etc.) ----------------
        if self.area_vao:
            self._set_mesh_uniforms(vp, camera.position, has_windows=False)
            ctx.disable(moderngl.CULL_FACE)
            self.mesh_prog['u_selected'].value = 0.0
            self.area_vao.render()
            ctx.enable(moderngl.CULL_FACE)

        # ---- 3. Buildings -----------------------------------------
        if self.building_vao:
            self._set_mesh_uniforms(vp, camera.position, has_windows=True)
            self.mesh_prog['u_selected'].value = 0.0
            self.building_vao.render()

        # ---- 4. Roads ---------------------------------------------
        if self.road_vao:
            self._set_mesh_uniforms(vp, camera.position, has_windows=False)
            self.road_vao.render()

        # ---- 5. Building interaction (point to inspect, grab to move) --
        aspect = w / h
        if gesture.get('point_uv'):
            u, v = gesture['point_uv']
            self._update_selection(u, v, camera, aspect)

        self._handle_grab(gesture, camera, aspect)

        # ---- 6. HUD overlay ---------------------------------------
        self._draw_hud(gesture, camera)
        self._draw_input_status(gesture)

    # ------------------------------------------------------------------
    #  Setup helpers
    # ------------------------------------------------------------------

    def _setup_sky(self):
        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
        vbo  = self.ctx.buffer(quad.tobytes())
        self.sky_vao = self.ctx.vertex_array(
            self.sky_prog, [(vbo, '2f', 'in_pos')])

    def _setup_bg(self):
        # Full-screen quad, UV flipped vertically (OpenGL vs OpenCV origin)
        quad = np.array([
            -1, -1,  0, 1,
             1, -1,  1, 1,
            -1,  1,  0, 0,
             1,  1,  1, 0,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self.bg_vao = self.ctx.vertex_array(
            self.bg_prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])

    def _setup_ground(self):
        extent = 2000.0
        gverts = np.array([
            [-extent, 0, -extent],
            [ extent, 0, -extent],
            [ extent, 0,  extent],
            [-extent, 0,  extent],
        ], np.float32)
        gidx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
        vbo  = self.ctx.buffer(gverts.tobytes())
        ibo  = self.ctx.buffer(gidx.tobytes())
        self.ground_vao = self.ctx.vertex_array(
            self.ground_prog, [(vbo, '3f', 'in_pos')], ibo)

    def _setup_hud(self):
        # Top-left corner quad for info panel (wider for metrics)
        quad = np.array([
            -1.0,  0.45,  0, 1,
            -0.35, 0.45,  1, 1,
            -1.0,  1.0,   0, 0,
            -0.35, 1.0,   1, 0,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        self.hud_vao = self.ctx.vertex_array(
            self.hud_prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])
        self.hud_texture = None

        # Top-right corner quad for input status
        quad_right = np.array([
            0.35,  0.45,  0, 1,
             1.0,  0.45,  1, 1,
            0.35,  1.0,   0, 0,
             1.0,  1.0,   1, 0,
        ], dtype=np.float32)
        vbo_right = self.ctx.buffer(quad_right.tobytes())
        self.input_hud_vao = self.ctx.vertex_array(
            self.hud_prog, [(vbo_right, '2f 2f', 'in_pos', 'in_uv')])
        self.input_hud_texture = None

    def _setup_webcam_texture(self):
        w, h = self.win_size
        self.cam_texture = self.ctx.texture((w, h), 3)
        self.cam_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ------------------------------------------------------------------
    #  Draw helpers
    # ------------------------------------------------------------------

    def _draw_background(self, frame: np.ndarray, gesture: dict):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)

        # Draw landmarks on frame
        frame_with_landmarks = frame.copy()
        landmarks = gesture.get('landmarks', [])
        for hand_landmarks in landmarks:
            self._draw_hand_landmarks(frame_with_landmarks, hand_landmarks)

        # Resize to window size and upload
        w, h   = self.win_size
        resized = cv2.resize(frame_with_landmarks, (w, h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.cam_texture.write(rgb.tobytes())
        self.cam_texture.use(location=0)
        self.bg_prog['u_tex'].value = 0
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)

        ctx.enable(moderngl.DEPTH_TEST)

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_landmarks):
        """Draw MediaPipe hand landmarks on the frame."""
        h, w, _ = frame.shape
        
        # Define connections between landmarks (MediaPipe hand connections)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (5, 9), (9, 13), (13, 17)  # palm connections
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(frame, start_point, end_point, (160, 10, 60), 2)
        
        # Draw landmarks
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (255, 225, 150), -1)

    def _draw_sky(self):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sky_prog['u_ambient'].value = self.ambient
        self.sky_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _set_mesh_uniforms(self, vp, cam_pos, has_windows=True, ambient_override=None):
        self.mesh_prog['u_vp'].write(vp.tobytes())
        self.mesh_prog['u_cam_pos'].write(cam_pos.astype(np.float32).tobytes())
        self.mesh_prog['u_sun_dir'].write(self.sun_dir.tobytes())
        self.mesh_prog['u_sun_color'].write(self.sun_color.tobytes())
        self.mesh_prog['u_ambient'].value = (ambient_override if ambient_override
                                             else self.ambient)
        self.mesh_prog['u_has_windows'].value = 1.0 if has_windows else 0.0

    def _update_selection(self, u: float, v: float, camera, aspect: float):
        origin, direction = camera.unproject_ray(u, v, aspect)
        best_idx  = -1
        best_dist = float('inf')

        for i, aabb in enumerate(self.building_aabbs):
            # Ray-AABB intersection (ground Y=0 to Y=height)
            t = _ray_aabb(origin, direction, aabb)
            if t is not None and t < best_dist:
                best_dist = t
                best_idx  = i

        if best_idx != self.selected_idx:
            self.selected_idx = best_idx
            self._build_hud_texture(best_idx)

    # ------------------------------------------------------------------
    #  Building grab-and-move
    # ------------------------------------------------------------------

    def _handle_grab(self, gesture: dict, camera, aspect: float):
        """Process grab gesture: pick building on grab start, move while held, release."""
        grab = gesture.get('grab', False)
        grab_uv = gesture.get('grab_uv')

        if grab and not self._grab_active and grab_uv:
            # Fist just closed — try to pick a building
            u, v = grab_uv
            origin, direction = camera.unproject_ray(u, v, aspect)
            best_idx = -1
            best_dist = float('inf')
            for i, aabb in enumerate(self.building_aabbs):
                t = _ray_aabb(origin, direction, aabb)
                if t is not None and t < best_dist:
                    best_dist = t
                    best_idx = i
            self._grabbed_idx = best_idx
            self._grab_active = True
            if best_idx >= 0:
                self._update_selection_direct(best_idx)

        elif grab and self._grab_active and self._grabbed_idx >= 0:
            # Fist still held — move the grabbed building
            pvx, pvz = gesture.get('pan_vel', (0.0, 0.0))
            if abs(pvx) > 1e-6 or abs(pvz) > 1e-6:
                # Convert normalised hand delta to world-space displacement
                scale = camera.distance * 2.0
                dx_world = pvx * scale
                dz_world = pvz * scale
                self._move_building(self._grabbed_idx, dx_world, dz_world)

        elif not grab and self._grab_active:
            # Fist released
            self._grab_active = False
            self._grabbed_idx = -1

    def _move_building(self, idx: int, dx: float, dz: float):
        """Translate a building's vertices and AABB by (dx, 0, dz)."""
        if self._scene_verts is None or idx >= len(self._vert_ranges):
            return
        v_start, v_end = self._vert_ranges[idx]
        if v_start >= v_end:
            return

        # Move vertices (column 0 = x, column 2 = z)
        self._scene_verts[v_start:v_end, 0] += dx
        self._scene_verts[v_start:v_end, 2] += dz

        # Update AABB
        aabb = self.building_aabbs[idx]
        aabb['min_x'] += dx
        aabb['max_x'] += dx
        aabb['min_z'] += dz
        aabb['max_z'] += dz

        # Re-upload the modified region to GPU
        byte_start = v_start * self._scene_verts.shape[1] * 4  # 4 bytes per float32
        byte_end   = v_end   * self._scene_verts.shape[1] * 4
        self._building_vbo.write(
            self._scene_verts[v_start:v_end].tobytes(),
            offset=byte_start,
        )

        # Update HUD to show new position
        self._build_hud_texture(idx)

    def _update_selection_direct(self, idx: int):
        """Select a building by index without ray-casting."""
        if idx != self.selected_idx:
            self.selected_idx = idx
            self._build_hud_texture(idx)

    def _build_hud_texture(self, idx: int):
        if idx < 0:
            self.hud_texture = None
            return

        aabb  = self.building_aabbs[idx]
        name  = aabb.get('name', '').strip()
        title = name if name else f"Building #{idx}"

        # Compute planning metrics
        metrics = self._compute_metrics()

        lines = [
            title,
            f"Height: {aabb['height']:.0f} m  (~{aabb['height']/3.5:.0f} floors)",
            f"W: {aabb['max_x']-aabb['min_x']:.0f} m  D: {aabb['max_z']-aabb['min_z']:.0f} m",
            "",
            f"Buildings: {metrics['count']}   Avg height: {metrics['avg_height']:.0f} m",
            f"Footprint: {metrics['footprint_pct']:.1f}%   Density: {metrics['density']:.1f}/ha",
        ]
        grabbed = " [GRABBED]" if self._grab_active and self._grabbed_idx == idx else ""
        lines[0] = title + grabbed

        img_h = 8 + len(lines) * 22 + 8
        img = Image.new('RGBA', (380, img_h), (20, 20, 30, 210))
        draw = ImageDraw.Draw(img)
        outline_col = (255, 180, 60, 255) if self._grab_active else (100, 180, 255, 200)
        draw.rectangle([0, 0, 379, img_h - 1], outline=outline_col, width=2)
        y = 8
        for i, line in enumerate(lines):
            if i == 0:
                color = (255, 200, 80) if self._grab_active else (100, 200, 255)
            elif i >= 4:
                color = (180, 220, 160)
            else:
                color = (220, 220, 220)
            draw.text((10, y), line, fill=color)
            y += 22

        arr = np.array(img.convert('RGB'), np.uint8)
        size = (380, img_h)
        if self.hud_texture is None or self.hud_texture.size != size:
            self.hud_texture = self.ctx.texture(size, 3)
        self.hud_texture.write(arr.tobytes())

    def _compute_metrics(self) -> dict:
        """Compute planning metrics from current building positions."""
        aabbs = self.building_aabbs
        count = len(aabbs)
        if count == 0:
            return {'count': 0, 'avg_height': 0, 'footprint_pct': 0, 'density': 0}

        total_height = sum(a['height'] for a in aabbs)
        total_footprint = sum(
            (a['max_x'] - a['min_x']) * (a['max_z'] - a['min_z']) for a in aabbs
        )

        # Compute bounding area of all buildings
        all_min_x = min(a['min_x'] for a in aabbs)
        all_max_x = max(a['max_x'] for a in aabbs)
        all_min_z = min(a['min_z'] for a in aabbs)
        all_max_z = max(a['max_z'] for a in aabbs)
        site_area = max((all_max_x - all_min_x) * (all_max_z - all_min_z), 1.0)

        return {
            'count': count,
            'avg_height': total_height / count,
            'footprint_pct': total_footprint / site_area * 100,
            'density': count / (site_area / 10000),  # buildings per hectare
        }

    def _draw_hud(self, gesture: dict, camera):
        if self.hud_texture is None:
            return
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.hud_texture.use(location=0)
        self.hud_prog['u_tex'].value = 0
        self.hud_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def _draw_input_status(self, gesture: dict):
        self._build_input_status_texture(gesture)
        if self.input_hud_texture is None:
            return
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.input_hud_texture.use(location=0)
        self.hud_prog['u_tex'].value = 0
        self.input_hud_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def _build_input_status_texture(self, gesture: dict):
        """Build texture showing current input mode and gesture status."""
        mode = gesture.get('mode', 'idle')
        hands = gesture.get('hands', 0)
        grab = gesture.get('grab', False)
        point_uv = gesture.get('point_uv')
        clench = gesture.get('clench', 0.0)
        
        # Determine status text
        if mode == 'idle':
            status = "IDLE"
            color = (150, 150, 150)
        elif grab:
            status = "GRABBING"
            color = (255, 100, 100)
        elif point_uv:
            status = "POINTING"
            color = (100, 255, 100)
        elif hands == 2:
            status = "ZOOM/ORBIT"
            color = (100, 100, 255)
        else:
            status = "PANNING"
            color = (255, 255, 100)
        
        lines = [
            f"Input: {status}",
            f"Hands: {hands}",
            f"Clench: {clench:.2f}" if clench > 0.1 else "",
        ]
        lines = [line for line in lines if line]  # Remove empty lines
        
        img_h = 8 + len(lines) * 22 + 8
        img = Image.new('RGBA', (300, img_h), (20, 20, 30, 210))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 299, img_h - 1], outline=(100, 180, 255, 200), width=2)
        y = 8
        for i, line in enumerate(lines):
            if i == 0:
                draw.text((10, y), line, fill=color)
            else:
                draw.text((10, y), line, fill=(220, 220, 220))
            y += 22

        arr = np.array(img.convert('RGB'), np.uint8)
        size = (300, img_h)
        if self.input_hud_texture is None or self.input_hud_texture.size != size:
            self.input_hud_texture = self.ctx.texture(size, 3)
        self.input_hud_texture.write(arr.tobytes())

    def resize(self, w: int, h: int):
        self.win_size = (w, h)
        self.cam_texture = self.ctx.texture((w, h), 3)
        self.cam_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)


# ------------------------------------------------------------------
#  Ray-AABB intersection
# ------------------------------------------------------------------

def _compute_vert_ranges(indices, verts, aabbs):
    """
    Figure out which vertex range belongs to each building.

    Each building's AABB has min_x/max_x/min_z/max_z. We assign each vertex
    to the building whose AABB contains it (with a small epsilon).
    Returns [(start_vert, end_vert), ...] per building.

    This works because _build_scene appends buildings sequentially, so vertices
    for building i come in a contiguous block before building i+1.
    """
    if len(aabbs) == 0 or len(verts) == 0:
        return []

    n_verts = len(verts)
    n_bldg  = len(aabbs)

    # Since buildings are added sequentially, vertex indices are monotonically
    # increasing per building. We can find boundaries by walking through the
    # index buffer and checking when vertices jump to a new AABB region.
    # Simpler approach: divide vertices evenly is wrong. Instead, use the fact
    # that gis_loader tracks v_off. We can reconstruct ranges from the vertices
    # themselves: find contiguous blocks where x,z are within each AABB.
    #
    # Fastest approach: since we know buildings were appended in order,
    # walk through verts and match to AABBs in order.
    ranges = []
    vi = 0
    for aabb in aabbs:
        start = vi
        eps = 0.5
        while vi < n_verts:
            x, z = float(verts[vi, 0]), float(verts[vi, 2])
            if (aabb['min_x'] - eps <= x <= aabb['max_x'] + eps and
                    aabb['min_z'] - eps <= z <= aabb['max_z'] + eps):
                vi += 1
            else:
                break
        ranges.append((start, vi))
    return ranges


def _ray_aabb(origin, direction, aabb) -> float | None:
    """Returns t (distance along ray) or None if no hit."""
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

    def slab(lo, hi, o, d):
        if abs(d) < 1e-9:
            return (-float('inf'), float('inf')) if lo <= o <= hi else (float('inf'), -float('inf'))
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        return (min(t1, t2), max(t1, t2))

    txmin, txmax = slab(aabb['min_x'], aabb['max_x'], ox, dx)
    tymin, tymax = slab(0.0,           aabb['height'], oy, dy)
    tzmin, tzmax = slab(aabb['min_z'], aabb['max_z'],  oz, dz)

    tmin = max(txmin, tymin, tzmin)
    tmax = min(txmax, tymax, tzmax)

    if tmax < 0 or tmin > tmax:
        return None
    return tmin if tmin >= 0 else tmax
