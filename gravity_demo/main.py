"""Entry point for the gravity demo."""

from __future__ import annotations

from .physics import Simulation
from .renderer import Renderer


def main() -> None:
    sim = Simulation()
    renderer = Renderer(sim)
    renderer.run()


if __name__ == "__main__":
    main()
