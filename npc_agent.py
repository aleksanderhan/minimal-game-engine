from __future__ import annotations

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from panda3d.core import TextNode

from object_manager import DynamicObject
from npc_brain import NPCBrain
from npc_command import NPCCommand
from npc_controller import NPCMovementController
from npc_perception import NPCPerception
from npc_world import WorldView


class NPCAgent:
    def __init__(
        self,
        game_engine,
        dynamic_object: DynamicObject,
        perception: NPCPerception,
        brain: NPCBrain,
        controller: NPCMovementController,
    ):
        self.game_engine = game_engine
        self.dynamic_object = dynamic_object
        self.perception = perception
        self.brain = brain
        self.controller = controller

        self.current_command = NPCCommand.stop()
        self.last_player_message = ""

        self.dialogue_text = OnscreenText(
            text="",
            pos=(-1.3, 0.78),
            scale=0.045,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
        )

    def receive_player_message(self, message: str) -> None:
        message = message.strip()

        if not message:
            return

        self.last_player_message = message

        world = self.build_world_view()

        command = self.brain.decide(
            player_message=message,
            world=world,
        )

        self.current_command = command

        if command.action == "respond_text":
            self.show_dialogue_text(f"NPC: {command.text}")
        else:
            self.show_dialogue_text(f"Player: {message}")

        self.execute_command(command, world)

    def update(self, task: Task) -> int:
        _ = globalClock.getDt()

        world = self.build_world_view()

        self.controller.apply_command(
            command=self.current_command,
            world=world,
        )

        return Task.cont

    def build_world_view(self) -> WorldView:
        return self.perception.build_world_view(
            npc_object=self.dynamic_object,
            player_node=self.game_engine.camera,
        )

    def execute_command(
        self,
        command: NPCCommand,
        world: WorldView,
    ) -> None:
        self.controller.apply_command(
            command=command,
            world=world,
        )

    def show_dialogue_text(self, text: str) -> None:
        self.dialogue_text.setText(text)