from __future__ import annotations

import math

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from panda3d.core import Quat, TextNode, Vec3

from constants import VoxelType
from npc_model import NPCBrain, NPCPolicyOutput
from voxel import create_dynamic_single_voxel_object


OBS_DIM = 12


class NPCAgent:
    def __init__(
        self,
        game_engine,
        brain: NPCBrain,
        position: Vec3,
        ai_hz: float,
    ):
        self.game_engine = game_engine
        self.brain = brain
        self.ai_interval = 1.0 / ai_hz
        self.time_since_ai_update = 0.0
        self.max_speed = 3.0 * self.game_engine.voxel_size

        self.object = create_dynamic_single_voxel_object(
            self.game_engine.voxel_size,
            VoxelType.STONE,
            self.game_engine.args.debug,
        )

        self.game_engine.object_manager.register_object(
            self.object,
            position,
            Vec3(0, 0, 0),
            Quat.identQuat(),
        )

        self.current_policy_output = NPCPolicyOutput(
            action_name="idle",
            speed=0.0,
            strafe=0.0,
            jump=0.0,
            should_speak=False,
            utterance="",
        )

        self.dialogue_text = OnscreenText(
            text="",
            pos=(-1.3, 0.78),
            scale=0.045,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
        )

    def update(self, task: Task) -> int:
        dt = globalClock.getDt()

        self.time_since_ai_update += dt
        if self.time_since_ai_update >= self.ai_interval:
            self.time_since_ai_update = 0.0
            observation = self._collect_observation()
            prompt = self._make_prompt(observation)
            self.current_policy_output = self.brain.act(observation, prompt)
            self._update_dialogue(self.current_policy_output)

        self._apply_policy_output(self.current_policy_output)

        return Task.cont

    def _collect_observation(self) -> np.ndarray:
        npc_pos = self.object.get_position()
        npc_velocity = self.object.get_velocity()
        player_pos = self.game_engine.camera.getPos()

        to_player = player_pos - npc_pos
        flat_to_player = Vec3(to_player.x, to_player.y, 0.0)
        distance_to_player = max(flat_to_player.length(), 1e-6)

        direction_to_player = flat_to_player / distance_to_player

        return np.array(
            [
                npc_pos.x,
                npc_pos.y,
                npc_pos.z,
                npc_velocity.x,
                npc_velocity.y,
                npc_velocity.z,
                player_pos.x,
                player_pos.y,
                player_pos.z,
                direction_to_player.x,
                direction_to_player.y,
                distance_to_player,
            ],
            dtype=np.float32,
        )

    def _make_prompt(self, observation: np.ndarray) -> str:
        (
            npc_x,
            npc_y,
            npc_z,
            vel_x,
            vel_y,
            vel_z,
            player_x,
            player_y,
            player_z,
            dir_x,
            dir_y,
            distance,
        ) = observation.tolist()

        return f"""
You are a voxel NPC in a minimal physics sandbox game.

NPC state:
- position: ({npc_x:.2f}, {npc_y:.2f}, {npc_z:.2f})
- velocity: ({vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f})

Player state:
- position: ({player_x:.2f}, {player_y:.2f}, {player_z:.2f})
- direction_to_player: ({dir_x:.2f}, {dir_y:.2f})
- distance_to_player: {distance:.2f}

Choose one physical action and optionally speak.
""".strip()

    def _apply_policy_output(self, policy_output: NPCPolicyOutput):
        npc_pos = self.object.get_position()
        player_pos = self.game_engine.camera.getPos()

        to_player = player_pos - npc_pos
        flat_to_player = Vec3(to_player.x, to_player.y, 0.0)

        if flat_to_player.length() > 1e-6:
            forward = flat_to_player.normalized()
        else:
            forward = Vec3(0, 1, 0)

        right = Vec3(forward.y, -forward.x, 0)

        movement = Vec3(0, 0, 0)

        if policy_output.action_name == "approach_player":
            movement += forward
        elif policy_output.action_name == "back_away":
            movement -= forward
        elif policy_output.action_name == "strafe_left":
            movement -= right
        elif policy_output.action_name == "strafe_right":
            movement += right
        elif policy_output.action_name == "jump":
            movement += Vec3(0, 0, policy_output.jump)

        movement += right * policy_output.strafe

        if movement.length() > 1e-6:
            movement.normalize()

        velocity = movement * self.max_speed * policy_output.speed

        current_velocity = self.object.get_velocity()
        self.object.set_velocity(
            Vec3(
                velocity.x,
                velocity.y,
                max(current_velocity.z, velocity.z),
            )
        )

    def _update_dialogue(self, policy_output: NPCPolicyOutput):
        if policy_output.should_speak and policy_output.utterance:
            self.dialogue_text.setText(f"NPC: {policy_output.utterance}")
        else:
            self.dialogue_text.setText(f"NPC action: {policy_output.action_name}")