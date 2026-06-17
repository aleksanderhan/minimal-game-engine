from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ARCHITECTURE_NAME = "npc_four_action_policy_v1"
ENCODER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DECODER_MODEL_NAME = "Qwen/Qwen3.5-2B"

ACTION_LABELS = (
    "move_towards_player",
    "move_away_from_player",
    "stop",
    "respond_text",
)

WORLD_STATE_DIM = 9
POLICY_DIM = 512
DEFAULT_POLICY_MAX_LENGTH = 192

APPROACH_STOP_DISTANCE = 2.0
BACK_AWAY_STOP_DISTANCE = 6.0


def policy_system_text() -> str:
    return """
System:
You control a friendly voxel NPC in a minimal physics sandbox game.
Choose exactly one action from this fixed set:
- move_towards_player: walk toward the player.
- move_away_from_player: walk away from the player.
- stop: stop moving and stay still.
- respond_text: do not move; answer the player in text.

Use only the latest player message and the current world state. No previous-command state is used; no stale-command recovery, object interaction, or hidden command filter exists.
""".strip()


def distance_band(distance: float) -> str:
    if distance < APPROACH_STOP_DISTANCE:
        return "very_close"
    if distance < 4.0:
        return "near"
    if distance < BACK_AWAY_STOP_DISTANCE:
        return "medium"
    return "far"


def make_policy_text(
    latest_player_text: str,
    distance: float,
    npc_speed: float,
) -> str:
    message = latest_player_text.strip() or "[no player message]"
    return f"""
{policy_system_text()}

Latest player message:
{message}

World state:
- distance_to_player: {distance:.2f}
- distance_band: {distance_band(distance)}
- npc_is_moving: {npc_speed > 0.15}
""".strip()


@dataclass(frozen=True)
class NPCPolicyOutput:
    action: str
    action_index: int
    action_confidence: float
    move_local: np.ndarray
    should_speak: bool
    utterance: str


def make_stop_policy_output() -> NPCPolicyOutput:
    return NPCPolicyOutput(
        action="stop",
        action_index=ACTION_LABELS.index("stop"),
        action_confidence=1.0,
        move_local=np.zeros(3, dtype=np.float32),
        should_speak=False,
        utterance="",
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance.to(x.dtype) + self.eps)
        return normalized * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        self.input = nn.Linear(input_dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, activation = self.input(x).chunk(2, dim=-1)
        return self.output(value * torch.nn.functional.silu(activation))


class ResidualSwiGLUBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ffn = SwiGLUFeedForward(dim, hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            RMSNorm(output_dim),
            nn.SiLU(),
            ResidualSwiGLUBlock(output_dim, hidden_dim),
            RMSNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(hidden_dim))
        self.norm = RMSNorm(hidden_dim)
        nn.init.normal_(self.query, mean=0.0, std=hidden_dim ** -0.5)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden.float())
        mask = attention_mask.bool()
        scores = torch.matmul(hidden, self.query) / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)


class NPCPolicyModel(nn.Module):
    def __init__(
        self,
        encoder_model_name: str,
        device: str,
        world_state_dim: int = WORLD_STATE_DIM,
        policy_dim: int = POLICY_DIM,
    ):
        super().__init__()
        self.encoder_model_name = encoder_model_name
        self.device_name = device
        self.runtime_device = torch.device(device)
        self.world_state_dim = world_state_dim
        self.policy_dim = policy_dim

        self.encoder = AutoModel.from_pretrained(
            encoder_model_name,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(self.runtime_device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_hidden_dim = int(self.encoder.config.hidden_size)
        hidden_dim = policy_dim * 2

        self.text_pool = MaskedAttentionPool(encoder_hidden_dim).to(self.runtime_device)
        self.text_projector = FeatureEncoder(
            input_dim=encoder_hidden_dim,
            output_dim=policy_dim,
            hidden_dim=hidden_dim,
        ).to(self.runtime_device)
        self.world_encoder = FeatureEncoder(
            input_dim=world_state_dim,
            output_dim=policy_dim,
            hidden_dim=hidden_dim,
        ).to(self.runtime_device)
        self.policy_backbone = nn.Sequential(
            nn.Linear(policy_dim * 2, policy_dim),
            RMSNorm(policy_dim),
            nn.SiLU(),
            ResidualSwiGLUBlock(policy_dim, hidden_dim),
            RMSNorm(policy_dim),
        ).to(self.runtime_device)
        self.action_head = nn.Linear(policy_dim, len(ACTION_LABELS)).to(self.runtime_device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.runtime_device)
        attention_mask = attention_mask.to(self.runtime_device)

        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        pooled = self.text_pool(outputs.last_hidden_state, attention_mask)
        return self.text_projector(pooled)

    def forward_encoded(
        self,
        policy_text: torch.Tensor,
        world_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        policy_text = policy_text.to(self.runtime_device).float()
        world_state = world_state.to(self.runtime_device).float()
        world = self.world_encoder(world_state)
        features = self.policy_backbone(torch.cat([policy_text, world], dim=-1))
        return {"action_logits": self.action_head(features)}

    def forward(
        self,
        policy_input_ids: torch.Tensor,
        policy_attention_mask: torch.Tensor,
        world_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        policy_text = self.encode_text(
            input_ids=policy_input_ids,
            attention_mask=policy_attention_mask,
        )
        return self.forward_encoded(policy_text=policy_text, world_state=world_state)


class NPCSpeechDecoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        use_4bit: bool,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        device_map = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if device == "cuda":
                device_map = {"": 0}
            elif device.startswith("cuda:"):
                device_map = {"": int(device.split(":", 1)[1])}
            else:
                device_map = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        if not use_4bit:
            self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        recent_dialogue_text: str,
        latest_player_text: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly voxel NPC in a minimal physics sandbox game. "
                    "Reply naturally and briefly. Do not mention coordinates, hidden state, "
                    "policy labels, model internals, or controls."
                ),
            },
            {
                "role": "user",
                "content": f"""
Recent dialogue:
{recent_dialogue_text}

Latest player message:
{latest_player_text}

Write only the NPC's spoken reply. Use at most one short sentence.
""".strip(),
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        input_len = encoded["input_ids"].shape[-1]
        generated = self.model.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
        return self._clean_utterance(response)

    def _clean_utterance(self, utterance: str) -> str:
        utterance = utterance.split("\n")[0].strip()
        utterance = utterance.removeprefix("NPC:").strip()
        utterance = utterance.removeprefix("NPC says:").strip()
        utterance = utterance.strip('"“”')
        words = utterance.split()
        utterance = " ".join(words[:14])
        if any(character.isdigit() for character in utterance):
            return ""
        return utterance


class NPCBrain:
    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        encoder_model_name: str = ENCODER_MODEL_NAME,
        decoder_model_name: str = DECODER_MODEL_NAME,
        speech_device: str | None = None,
        use_4bit_decoder: bool = True,
    ):
        self.device = torch.device(device)
        self.speech_device = speech_device or device

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        architecture = checkpoint.get("architecture")
        if architecture != ARCHITECTURE_NAME:
            raise RuntimeError(
                f"Checkpoint architecture {architecture!r} does not match {ARCHITECTURE_NAME!r}"
            )

        self.encoder_model_name = checkpoint.get("encoder_model_name", encoder_model_name)
        self.decoder_model_name = checkpoint.get("decoder_model_name", decoder_model_name)
        self.policy_max_length = int(checkpoint.get("policy_max_length", DEFAULT_POLICY_MAX_LENGTH))
        self.world_state_dim = int(checkpoint.get("world_state_dim", WORLD_STATE_DIM))
        self.policy_dim = int(checkpoint.get("policy_dim", POLICY_DIM))
        self.action_labels = tuple(checkpoint.get("action_labels", ACTION_LABELS))

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name)
        self.policy = NPCPolicyModel(
            encoder_model_name=self.encoder_model_name,
            device=device,
            world_state_dim=self.world_state_dim,
            policy_dim=self.policy_dim,
        )
        self.policy.load_state_dict(checkpoint["policy_model"], strict=True)
        del checkpoint

        self.policy.eval()
        self.policy.encoder.eval()
        self.speech_decoder = NPCSpeechDecoder(
            model_name=self.decoder_model_name,
            device=self.speech_device,
            use_4bit=use_4bit_decoder,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.encoder_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.policy_max_length,
        )
        return {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }

    def _world_state_from_observation(self, observation: np.ndarray) -> torch.Tensor:
        obs = observation.astype(np.float32)
        npc_pos = obs[0:3]
        npc_velocity = obs[3:6]
        player_pos = obs[6:9]
        direction_to_player = obs[9:11]
        distance = obs[11:12]
        relative_player_pos = player_pos - npc_pos
        world_state = np.concatenate(
            [npc_velocity, relative_player_pos, direction_to_player, distance],
            axis=0,
        ).astype(np.float32)
        return torch.tensor(world_state, dtype=torch.float32, device=self.device).unsqueeze(0)

    @torch.inference_mode()
    def act(
        self,
        observation: np.ndarray,
        policy_text: str,
        recent_dialogue_text: str,
        latest_player_text: str,
    ) -> NPCPolicyOutput:
        encoded_policy = self._encode_texts([policy_text])
        output = self.policy(
            policy_input_ids=encoded_policy["input_ids"],
            policy_attention_mask=encoded_policy["attention_mask"],
            world_state=self._world_state_from_observation(observation),
        )

        probabilities = torch.softmax(output["action_logits"], dim=-1)[0]
        action_index = int(torch.argmax(probabilities).detach().cpu().item())
        action = self.action_labels[action_index]
        confidence = float(probabilities[action_index].detach().cpu().item())
        move_local = self._move_for_action(action=action, observation=observation)

        utterance = ""
        if action == "respond_text":
            utterance = self.speech_decoder.generate(
                recent_dialogue_text=recent_dialogue_text,
                latest_player_text=latest_player_text,
            )

        return NPCPolicyOutput(
            action=action,
            action_index=action_index,
            action_confidence=confidence,
            move_local=move_local,
            should_speak=bool(utterance),
            utterance=utterance,
        )

    def _move_for_action(self, action: str, observation: np.ndarray) -> np.ndarray:
        distance = float(observation.astype(np.float32)[11])
        move = np.zeros(3, dtype=np.float32)

        if action == "move_towards_player" and distance > APPROACH_STOP_DISTANCE:
            move[1] = np.clip((distance - APPROACH_STOP_DISTANCE) / 5.0, 0.25, 1.0)
        elif action == "move_away_from_player" and distance < BACK_AWAY_STOP_DISTANCE:
            move[1] = -np.clip((BACK_AWAY_STOP_DISTANCE - distance) / 4.0, 0.25, 1.0)

        return move.astype(np.float32)
