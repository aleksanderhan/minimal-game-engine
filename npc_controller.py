from __future__ import annotations

from panda3d.core import Point3, Vec3

from object_manager import DynamicObject
from npc_command import NPCCommand
from npc_world import TargetPositionResolver, WorldView


class NPCMovementController:
    def __init__(
        self,
        dynamic_object: DynamicObject,
        target_position_resolver: TargetPositionResolver,
        max_speed: float = 5.0,
        stop_distance: float = 1.5,
    ):
        self.dynamic_object = dynamic_object
        self.target_position_resolver = target_position_resolver
        self.max_speed = max_speed
        self.stop_distance = stop_distance

    def apply_command(
        self,
        command: NPCCommand,
        world: WorldView,
    ) -> None:
        if command.action == "stop":
            self.stop()
            return

        if command.action == "respond_text":
            self.stop()
            return

        target_position = self.target_position_resolver.resolve_position(
            target=command.target,
            world=world,
        )

        if target_position is None:
            self.stop()
            return

        if command.action == "move_towards":
            self.move_towards(target_position)
            return

        if command.action == "move_away":
            self.move_away(target_position)
            return

        raise ValueError(f"Unhandled NPC command action: {command.action}")

    def move_towards(self, target_position: Point3) -> None:
        direction, distance = self.compute_flat_direction_to(target_position)

        if distance <= self.stop_distance:
            self.stop()
            return

        self.set_horizontal_velocity(direction, self.max_speed)

    def move_away(self, target_position: Point3) -> None:
        direction, distance = self.compute_flat_direction_to(target_position)

        if distance <= 1.0e-6:
            self.stop()
            return

        self.set_horizontal_velocity(-direction, self.max_speed)

    def stop(self) -> None:
        current_velocity = self.dynamic_object.get_velocity()
        self.dynamic_object.set_velocity(
            Vec3(0.0, 0.0, current_velocity.z)
        )

    def compute_flat_direction_to(self, target_position: Point3) -> tuple[Vec3, float]:
        npc_position = self.dynamic_object.get_position()

        offset = target_position - npc_position
        flat_offset = Vec3(offset.x, offset.y, 0.0)
        distance = flat_offset.length()

        if distance <= 1.0e-6:
            return Vec3(0.0, 0.0, 0.0), 0.0

        return flat_offset / distance, distance

    def set_horizontal_velocity(
        self,
        direction: Vec3,
        speed: float,
    ) -> None:
        current_velocity = self.dynamic_object.get_velocity()

        self.dynamic_object.set_velocity(
            Vec3(
                direction.x * speed,
                direction.y * speed,
                current_velocity.z,
            )
        )