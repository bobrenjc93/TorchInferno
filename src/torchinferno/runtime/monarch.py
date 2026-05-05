from __future__ import annotations

from dataclasses import dataclass

from torchinferno.runtime.fake_dist import FakeProcessWorld


def monarch_available() -> bool:
    try:
        import monarch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class MonarchMeshSpec:
    world_size: int
    mesh_shape: tuple[int, ...]


class MonarchOrFakeWorld:
    """Adapter point for Monarch-backed execution with a local fake fallback."""

    def __init__(self, spec: MonarchMeshSpec) -> None:
        self.spec = spec
        self.fake_world = FakeProcessWorld(spec.world_size, mesh_shape=spec.mesh_shape)

    @property
    def using_monarch(self) -> bool:
        return monarch_available()

    def local_world(self) -> FakeProcessWorld:
        return self.fake_world
