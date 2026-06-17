from __future__ import annotations

from panda3d.core import NodePath, Vec3

from object_manager import DynamicObject, ObjectManager
from npc_world import ObjectView, WorldView


class NPCPerception:
    def __init__(
        self,
        object_manager: ObjectManager,
        max_visible_objects: int = 12,
    ):
        self.object_manager = object_manager
        self.max_visible_objects = max_visible_objects

    def build_world_view(
        self,
        npc_object: DynamicObject,
        player_node: NodePath,
    ) -> WorldView:
        npc_position = npc_object.get_position()
        npc_velocity = npc_object.get_velocity()
        player_position = player_node.getPos()

        object_views = []

        for dynamic_object in self.object_manager.get_objects_except(npc_object.id):
            object_view = self.build_object_view(
                npc_object=npc_object,
                dynamic_object=dynamic_object,
            )
            object_views.append(object_view)

        object_views = self.sort_objects_by_distance(object_views)
        object_views = self.filter_visible_objects(object_views)

        return WorldView(
            npc_position=npc_position,
            npc_velocity=npc_velocity,
            player_position=player_position,
            player_velocity=None,
            objects=object_views[: self.max_visible_objects],
        )

    def build_object_view(
        self,
        npc_object: DynamicObject,
        dynamic_object: DynamicObject,
    ) -> ObjectView:
        npc_position = npc_object.get_position()
        object_position = dynamic_object.get_position()
        object_velocity = dynamic_object.get_velocity()

        offset = object_position - npc_position
        flat_offset = Vec3(offset.x, offset.y, 0.0)
        distance = flat_offset.length()

        if distance > 1.0e-6:
            direction = flat_offset / distance
        else:
            direction = Vec3(0.0, 0.0, 0.0)

        return ObjectView(
            object_id=dynamic_object.id,
            name=dynamic_object.name,
            kind=dynamic_object.kind,
            color_name=dynamic_object.color_name,
            tags=set(dynamic_object.tags),
            position=object_position,
            velocity=object_velocity,
            distance_to_npc=distance,
            direction_from_npc=direction,
            direction_label=self.direction_label_from_vector(direction),
            is_visible=True,
            is_reachable=True,
        )

    def sort_objects_by_distance(
        self,
        objects: list[ObjectView],
    ) -> list[ObjectView]:
        return sorted(objects, key=lambda object_view: object_view.distance_to_npc)

    def filter_visible_objects(
        self,
        objects: list[ObjectView],
    ) -> list[ObjectView]:
        return objects

    def direction_label_from_vector(self, direction: Vec3) -> str:
        if direction.length() <= 1.0e-6:
            return "same_position"

        abs_x = abs(direction.x)
        abs_y = abs(direction.y)

        if abs_x > abs_y:
            return "right" if direction.x > 0.0 else "left"

        return "front" if direction.y > 0.0 else "behind"