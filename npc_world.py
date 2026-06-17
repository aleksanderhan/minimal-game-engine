from __future__ import annotations

from dataclasses import dataclass, field

from panda3d.core import Point3, Vec3

from npc_command import TargetRef


@dataclass(frozen=True)
class ObjectQuery:
    kind: str | None = None
    color_name: str | None = None
    name_contains: str | None = None
    required_tags: set[str] = field(default_factory=set)

    def matches(self, object_view: "ObjectView") -> bool:
        if self.kind is not None and object_view.kind != self.kind:
            return False

        if self.color_name is not None and object_view.color_name != self.color_name:
            return False

        if self.name_contains is not None:
            if self.name_contains.lower() not in object_view.name.lower():
                return False

        if not self.required_tags.issubset(object_view.tags):
            return False

        return True


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

    def objects_matching(self, query: ObjectQuery) -> list[ObjectView]:
        return [
            object_view
            for object_view in self.objects
            if query.matches(object_view)
        ]

    def nearest_matching_object(self, query: ObjectQuery) -> ObjectView | None:
        matches = self.objects_matching(query)

        if not matches:
            return None

        return min(matches, key=lambda object_view: object_view.distance_to_npc)


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