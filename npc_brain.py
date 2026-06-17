from __future__ import annotations

from npc_command import NPCAction, NPCCommand
from npc_target_resolver import NPCTargetResolver
from npc_world import WorldView


class NPCActionClassifier:
    def classify_action(
        self,
        player_message: str,
        world: WorldView,
    ) -> NPCAction:
        text = player_message.lower()

        if self._contains_any(text, ["stop", "freeze", "hold still", "stay there", "wait"]):
            return "stop"

        if self._contains_any(text, ["back away", "move away", "go away", "back up", "give me space"]):
            return "move_away"

        if self._contains_any(text, ["come here", "come to me", "move to me", "follow me"]):
            return "move_towards"

        if self._contains_any(text, ["go to", "move to", "walk to", "approach", "come to"]):
            return "move_towards"

        if self._contains_any(text, ["what", "where", "who", "why", "how", "say", "tell"]):
            return "respond_text"

        return "respond_text"

    def _contains_any(self, text: str, phrases: list[str]) -> bool:
        return any(phrase in text for phrase in phrases)


class NPCTextResponder:
    def make_response(
        self,
        player_message: str,
        world: WorldView,
    ) -> str:
        text = player_message.lower()

        if "see" in text or "around" in text or "nearby" in text:
            return self._describe_visible_objects(world)

        return "Okay."

    def _describe_visible_objects(self, world: WorldView) -> str:
        if not world.objects:
            return "I don't see any objects nearby."

        descriptions = [
            f"{object_view.color_name} {object_view.kind}"
            for object_view in world.objects[:5]
        ]

        return "I see " + ", ".join(descriptions) + "."


class NPCBrain:
    def __init__(
        self,
        action_classifier: NPCActionClassifier,
        text_responder: NPCTextResponder,
        target_resolver: NPCTargetResolver,
    ):
        self.action_classifier = action_classifier
        self.text_responder = text_responder
        self.target_resolver = target_resolver

    def decide(
        self,
        player_message: str,
        world: WorldView,
    ) -> NPCCommand:
        action = self.action_classifier.classify_action(
            player_message=player_message,
            world=world,
        )

        if action == "respond_text":
            response = self.text_responder.make_response(
                player_message=player_message,
                world=world,
            )
            return NPCCommand.respond_text(response)

        if action == "stop":
            return NPCCommand.stop()

        target = self.target_resolver.resolve_target(
            player_message=player_message,
            action=action,
            world=world,
        )

        if action == "move_towards":
            return NPCCommand.move_towards(target)

        if action == "move_away":
            return NPCCommand.move_away(target)

        raise ValueError(f"Unhandled NPC action: {action}")