"""Physics engine for the gravity demo."""

from __future__ import annotations

import numpy as np

G = 6.67430e-11


class Body:
    """A physical body."""

    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: float, color: np.ndarray) -> None:
        self.pos = pos.astype(float)
        self.vel = vel.astype(float)
        self.mass = float(mass)
        self.color = color.astype(float)
        self.trail = []


class Simulation:
    """Handles body integration and force computation."""

    def __init__(self) -> None:
        self.bodies: list[Body] = []
        self.alpha = 0.0
        self.lambda_s = 1e9
        self.use_screened = False
        self.paused = False
        self.dt = 1.0
        self.base_dt = 1.0

    def reset(self) -> None:
        self.bodies.clear()
        earth = Body(np.array([0.0, 0.0, 0.0]), np.zeros(3), 5.972e24, np.array([0.2, 0.3, 1.0]))
        moon_distance = 384400000.0
        moon_speed = np.sqrt(G * earth.mass / moon_distance)
        moon = Body(
            np.array([moon_distance, 0.0, 0.0]),
            np.array([0.0, moon_speed, 0.0]),
            7.342e22,
            np.array([0.6, 0.6, 0.6]),
        )
        probe = Body(
            np.array([moon_distance * 0.5, 0.0, 0.0]),
            np.zeros(3),
            1.0,
            np.array([1.0, 0.1, 0.1]),
        )
        self.bodies.extend([earth, moon, probe])

    def _force(self, r: np.ndarray, m1: float, m2: float) -> np.ndarray:
        dist = np.linalg.norm(r)
        if dist < 1e-3:
            return np.zeros(3)
        force = G * m1 * m2 / dist**3 * r
        if self.use_screened:
            force *= 1.0 + self.alpha * np.exp(-dist / self.lambda_s)
        return force

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        n = len(self.bodies)
        pos = y[:3 * n].reshape((n, 3))
        vel = y[3 * n:].reshape((n, 3))
        acc = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                r = pos[j] - pos[i]
                f = self._force(r, self.bodies[i].mass, self.bodies[j].mass)
                acc[i] += f / self.bodies[i].mass
                acc[j] -= f / self.bodies[j].mass
        return np.hstack([vel.reshape(-1), acc.reshape(-1)])

    def _rk4_step(self, dt: float) -> None:
        n = len(self.bodies)
        y0 = np.hstack([b.pos for b in self.bodies] + [b.vel for b in self.bodies])
        k1 = self._derivatives(y0)
        k2 = self._derivatives(y0 + 0.5 * dt * k1)
        k3 = self._derivatives(y0 + 0.5 * dt * k2)
        k4 = self._derivatives(y0 + dt * k3)
        y = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        for i, b in enumerate(self.bodies):
            b.pos = y[3 * i:3 * i + 3]
            b.vel = y[3 * n + 3 * i:3 * n + 3 * i + 3]
            b.trail.append(b.pos.copy())
            if len(b.trail) > 200:
                b.trail.pop(0)

    def _adapt_dt(self) -> None:
        """Adjust time step based on the fastest body."""
        speeds = [np.linalg.norm(b.vel) for b in self.bodies]
        max_speed = max(speeds) if speeds else 0.0
        if max_speed == 0:
            self.dt = self.base_dt
        else:
            self.dt = self.base_dt / (1.0 + max_speed / 1e4)

    def step(self) -> None:
        if self.paused:
            return
        self._adapt_dt()
        self._rk4_step(self.dt)

