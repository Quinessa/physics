"""Rendering utilities using pygame and moderngl."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import moderngl
import numpy as np
import pygame
from imgui_bundle import imgui, imgui_widgets
from pyrr import Matrix44, Vector3

from .physics import Body, Simulation


class Renderer:
    """OpenGL renderer with camera controls and GUI."""

    def __init__(self, sim: Simulation) -> None:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        self.screen = pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Gravity Demo")
        self.ctx = moderngl.create_context()
        self.sim = sim
        self.prog_sphere = self._load_program("sphere")
        self.prog_trail = self._load_program("trail")
        self.sphere_vao = self._create_sphere_vao()
        self.trail_vbo = self.ctx.buffer(reserve=3 * 200 * 4)
        self.camera_pos = Vector3([0.0, -1.0, 0.5]) * 5e8
        self.camera_yaw = 0.0
        self.camera_pitch = 0.3
        self.show_trails = True

    def _load_program(self, name: str) -> moderngl.Program:
        shader_dir = Path(__file__).parent / "assets" / "shaders"
        with open(shader_dir / f"{name}.vert") as f:
            vert = f.read()
        with open(shader_dir / f"{name}.frag") as f:
            frag = f.read()
        return self.ctx.program(vertex_shader=vert, fragment_shader=frag)

    def _create_sphere_vao(self) -> moderngl.VertexArray:
        lat, lon = 16, 16
        verts = []
        norms = []
        for i in range(lat + 1):
            theta = math.pi * i / lat
            for j in range(lon + 1):
                phi = 2 * math.pi * j / lon
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                verts.append((x, y, z))
                norms.append((x, y, z))
        indices = []
        for i in range(lat):
            for j in range(lon):
                i1 = i * (lon + 1) + j
                i2 = i1 + lon + 1
                indices += [i1, i2, i1 + 1, i1 + 1, i2, i2 + 1]
        vbo = self.ctx.buffer(np.array(verts, dtype="f4"))
        nbo = self.ctx.buffer(np.array(norms, dtype="f4"))
        ibo = self.ctx.buffer(np.array(indices, dtype="i4"))
        vao = self.ctx.vertex_array(
            self.prog_sphere,
            [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
            ibo,
        )
        self.sphere_indices = len(indices)
        return vao

    def _get_mvp(self, position: Vector3, radius: float) -> Matrix44:
        model = Matrix44.from_scale([radius] * 3) * Matrix44.from_translation(position)
        view = Matrix44.from_eulers([self.camera_pitch, 0.0, self.camera_yaw])
        view *= Matrix44.from_translation(-self.camera_pos)
        proj = Matrix44.perspective_projection(45.0, 1280 / 720, 1e6, 1e12)
        return proj * view * model

    def _process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.sim.use_screened = not self.sim.use_screened
                if event.key == pygame.K_r:
                    self.sim.reset()
                if event.key == pygame.K_SPACE:
                    self.sim.paused = not self.sim.paused
            if event.type == pygame.MOUSEWHEEL:
                self.camera_pos *= 0.9 if event.y > 0 else 1.1
        return True

    def _draw_body(self, body: Body) -> None:
        mvp = self._get_mvp(Vector3(body.pos), 0.1e9)
        self.prog_sphere["mvp"].write(mvp.astype("f4"))
        self.prog_sphere["model"].write(Matrix44.from_translation(body.pos).astype("f4"))
        self.prog_sphere["color"].value = tuple(body.color)
        self.sphere_vao.render(mode=moderngl.TRIANGLES, vertices=self.sphere_indices)

    def _draw_trail(self, body: Body) -> None:
        if not body.trail:
            return
        arr = np.array(body.trail, dtype="f4")
        self.trail_vbo.orphan(len(arr) * 12)
        self.trail_vbo.write(arr)
        self.prog_trail["mvp"].write(self._get_mvp(Vector3(), 1.0).astype("f4"))
        self.prog_trail["color"].value = tuple(body.color)
        vao = self.ctx.vertex_array(self.prog_trail, [(self.trail_vbo, "3f", "in_position")])
        vao.render(mode=moderngl.LINE_STRIP, vertices=len(arr))

    def _draw_gui(self) -> None:
        imgui.new_frame()
        imgui.begin("Parameters", True)
        changed, self.sim.alpha = imgui.slider_float("alpha", self.sim.alpha, -1.0, 1.0)
        changed, log_lambda = imgui.slider_float("lambda_s (log10)", math.log10(self.sim.lambda_s), 0.0, 9.0)
        self.sim.lambda_s = 10 ** log_lambda
        changed, self.show_trails = imgui.checkbox("show trails", self.show_trails)
        imgui.text("Paused" if self.sim.paused else "Running")
        eq = "F = G m1 m2 / r^2"
        if self.sim.use_screened:
            eq += " * (1 + alpha e^{-r/lambda_s})"
        imgui.text_colored(eq, 0.7, 0.7, 0.7)
        imgui.end()
        imgui.render()
        imgui_widgets.render(imgui.get_draw_data())

    def run(self) -> None:
        self.sim.reset()
        clock = pygame.time.Clock()
        running = True
        while running:
            running = self._process_events()
            self.sim.step()
            self.ctx.clear(0.07, 0.07, 0.08)
            for body in self.sim.bodies:
                self._draw_body(body)
                if self.show_trails:
                    self._draw_trail(body)
            self._draw_gui()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

