from __future__ import annotations

from dataclasses import dataclass

from panda3d.core import Point3, Vec3

from npc_command import TargetRef


@dataclass(frozen=True)
class ObjectView:
    object_id: str
    name: str
    kind: str
    color_name: str
    tags: set[str]

    position: Point3
    velocity: Vec3

    distance_to_npc: float
    direction_from_npc: Vec3
    direction_label: str

    is_visible: bool = True
    is_reachable: bool = True


@dataclass(frozen=True)
class WorldView:
    npc_position: Point3
    npc_velocity: Vec3

    player_position: Point3
    player_velocity: Vec3 | None

    objects: list[ObjectView]

    def nearest_object(self) -> ObjectView | None:
        if not self.objects:
            return None

        return min(self.objects, key=lambda object_view: object_view.distance_to_npc)


class TargetPositionResolver:
    def resolve_position(self, target: TargetRef, world: WorldView) -> Point3 | None:
        if target.kind == "player":
            return world.player_position

        if target.kind == "position":
            return target.position

        if target.kind == "object":
            return self._resolve_object_position(target, world)

        if target.kind == "none":
            return None

        raise ValueError(f"Unhandled target kind: {target.kind}")

    def _resolve_object_position(
        self,
        target: TargetRef,
        world: WorldView,
    ) -> Point3 | None:
        if target.object_id is None:
            return None

        for object_view in world.objects:
            if object_view.object_id == target.object_id:
                return object_view.position

        return None
