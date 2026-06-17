from __future__ import annotations

from npc_command import NPCAction, TargetRef
from npc_world import ObjectQuery, WorldView


COLOR_WORDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "white",
    "black",
    "gray",
    "grey",
    "orange",
    "purple",
}

KIND_WORDS = {
    "ball",
    "sphere",
    "cube",
    "box",
    "voxel",
    "object",
}


PLAYER_TARGET_WORDS = {
    "me",
    "myself",
    "player",
    "here",
}


class NPCTargetResolver:
    def resolve_target(
        self,
        player_message: str,
        action: NPCAction,
        world: WorldView,
    ) -> TargetRef:
        if action in {"stop", "respond_text"}:
            return TargetRef.none()

        player_target = self.resolve_player_target(player_message)
        if player_target is not None:
            return player_target

        object_target = self.resolve_object_target(player_message, world)
        if object_target is not None:
            return object_target

        return TargetRef.none()

    def resolve_player_target(self, player_message: str) -> TargetRef | None:
        words = self._words(player_message)

        if words.intersection(PLAYER_TARGET_WORDS):
            return TargetRef.player()

        if "come" in words and "here" in words:
            return TargetRef.player()

        return None

    def resolve_object_target(
        self,
        player_message: str,
        world: WorldView,
    ) -> TargetRef | None:
        query = self.parse_object_query(player_message)

        if query is None:
            return None

        object_view = world.nearest_matching_object(query)

        if object_view is None:
            return None

        return TargetRef.object(
            object_id=object_view.object_id,
            label=object_view.name,
        )

    def resolve_pointing_target(self, world: WorldView) -> TargetRef | None:
        nearest_object = world.nearest_object()

        if nearest_object is None:
            return None

        return TargetRef.object(
            object_id=nearest_object.object_id,
            label=nearest_object.name,
        )

    def parse_object_query(self, player_message: str) -> ObjectQuery | None:
        words = self._words(player_message)

        color_name = None
        kind = None

        for word in words:
            if word in COLOR_WORDS:
                color_name = "gray" if word == "grey" else word

            if word in KIND_WORDS:
                kind = self._normalize_kind(word)

        if color_name is None and kind is None:
            return None

        return ObjectQuery(
            kind=kind,
            color_name=color_name,
        )

    def _normalize_kind(self, word: str) -> str:
        if word == "sphere":
            return "ball"

        if word == "box":
            return "cube"

        if word == "object":
            return None

        return word

    def _words(self, text: str) -> set[str]:
        cleaned = text.lower()

        for char in ",.!?;:-_":
            cleaned = cleaned.replace(char, " ")

        return set(cleaned.split())