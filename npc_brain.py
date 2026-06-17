from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

from npc_command import NPCAction, NPCCommand, TargetRef
from npc_world import ObjectView, WorldView


ENCODER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DECODER_MODEL_NAME = "Qwen/Qwen3.5-2B"
DEFAULT_ENCODER_MAX_LENGTH = 128
DEFAULT_DECODER_MAX_LENGTH = 768


@dataclass(frozen=True)
class ActionSpec:
    action: NPCAction
    examples: tuple[str, ...]
    requires_target: bool
    threshold: float
    margin: float


@dataclass(frozen=True)
class ActionMatch:
    action: NPCAction
    score: float
    second_score: float


@dataclass(frozen=True)
class TargetCandidate:
    target: TargetRef
    examples: tuple[str, ...]


@dataclass(frozen=True)
class TargetMatch:
    target: TargetRef
    score: float
    second_score: float


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        action="move_towards",
        examples=(
            "walk toward the target",
            "go to the target",
            "move to the target",
            "approach the target",
            "come closer",
            "come here",
            "follow me",
            "check out the target",
            "inspect the target",
        ),
        requires_target=True,
        threshold=0.40,
        margin=0.03,
    ),
    ActionSpec(
        action="move_away",
        examples=(
            "move away from the target",
            "back away from the target",
            "retreat from the target",
            "give me space",
            "stay away from me",
            "do not come closer",
            "keep your distance",
        ),
        requires_target=True,
        threshold=0.40,
        margin=0.03,
    ),
    ActionSpec(
        action="stop",
        examples=(
            "stop moving",
            "stand still",
            "freeze in place",
            "hold position",
            "wait there",
            "stay there",
            "do nothing",
        ),
        requires_target=False,
        threshold=0.42,
        margin=0.03,
    ),
    ActionSpec(
        action="respond_text",
        examples=(
            "answer the player's question",
            "tell the player something",
            "say something",
            "describe what you can see",
            "what do you see",
            "what is nearby",
            "talk to me",
        ),
        requires_target=False,
        threshold=0.38,
        margin=0.03,
    ),
)


class NPCSemanticEncoder:
    def __init__(
        self,
        model_name: str = ENCODER_MODEL_NAME,
        device: str | None = None,
        max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        text_list = [text.strip() for text in texts if text.strip()]

        if not text_list:
            raise ValueError("Cannot encode an empty text list.")

        encoded = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }

        output = self.model(**encoded, return_dict=True)
        hidden = output.last_hidden_state.float()
        attention_mask = encoded["attention_mask"].float().unsqueeze(-1)

        pooled = (hidden * attention_mask).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, p=2, dim=-1)

    def cosine_scores(
        self,
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(candidate_embeddings, query_embedding.squeeze(0))


class NPCSpeechDecoder:
    def __init__(
        self,
        model_name: str = DECODER_MODEL_NAME,
        device: str | None = None,
        use_4bit: bool = True,
        max_length: int = DEFAULT_DECODER_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
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
            if self.device.type == "cuda" and self.device.index is not None:
                device_map = {"": self.device.index}
            elif self.device.type == "cuda":
                device_map = {"": 0}
            else:
                device_map = {"": str(self.device)}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        if not use_4bit:
            self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        context_text: str,
        latest_player_text: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly voxel NPC in a minimal physics sandbox game. "
                    "Use the visible-world context when it matters. Do not invent unseen objects. "
                    "Reply naturally and briefly. Do not mention coordinates, hidden state, "
                    "policy labels, model internals, or controls."
                ),
            },
            {
                "role": "user",
                "content": f"""
Visible-world context:
{context_text}

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
            max_length=self.max_length,
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


class SemanticCommandMapper:
    def __init__(
        self,
        encoder: NPCSemanticEncoder,
        action_specs: tuple[ActionSpec, ...] = ACTION_SPECS,
        target_threshold: float = 0.34,
        target_margin: float = 0.03,
        debug: bool = False,
    ):
        self.encoder = encoder
        self.action_specs = action_specs
        self.target_threshold = target_threshold
        self.target_margin = target_margin
        self.debug = debug

        self._action_examples: list[str] = []
        self._action_example_specs: list[ActionSpec] = []

        for spec in self.action_specs:
            for example in spec.examples:
                self._action_examples.append(example)
                self._action_example_specs.append(spec)

        self._action_embeddings = self.encoder.encode(self._action_examples)

    def map_message_to_command(
        self,
        player_message: str,
        world: WorldView,
    ) -> NPCCommand:
        message = player_message.strip()

        if not message:
            return NPCCommand.respond_text("I did not hear anything.")

        message_embedding = self.encoder.encode([message])
        action_match = self.match_action(message_embedding)

        if action_match is None:
            return NPCCommand.respond_text("I do not understand that command yet.")

        if action_match.action == "respond_text":
            return NPCCommand.respond_text("")

        if action_match.action == "stop":
            return NPCCommand.stop()

        target_match = self.match_target(
            message_embedding=message_embedding,
            world=world,
        )

        if target_match is None:
            return NPCCommand.respond_text("Which target?")

        if action_match.action == "move_towards":
            return NPCCommand.move_towards(target_match.target)

        if action_match.action == "move_away":
            return NPCCommand.move_away(target_match.target)

        raise ValueError(f"Unhandled semantic action: {action_match.action}")

    def match_action(
        self,
        message_embedding: torch.Tensor,
    ) -> ActionMatch | None:
        scores = self.encoder.cosine_scores(
            query_embedding=message_embedding,
            candidate_embeddings=self._action_embeddings,
        )

        action_scores: dict[NPCAction, float] = {
            spec.action: float("-inf")
            for spec in self.action_specs
        }

        for score, spec in zip(scores.detach().cpu().tolist(), self._action_example_specs):
            action_scores[spec.action] = max(action_scores[spec.action], float(score))

        ranked = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        top_action, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
        top_spec = self._spec_for_action(top_action)

        if self.debug:
            self._print_ranked_scores("Action scores", ranked)

        if top_score < top_spec.threshold:
            return None

        if top_score - second_score < top_spec.margin:
            return None

        return ActionMatch(
            action=top_action,
            score=top_score,
            second_score=second_score,
        )

    def match_target(
        self,
        message_embedding: torch.Tensor,
        world: WorldView,
    ) -> TargetMatch | None:
        candidates = self.build_target_candidates(world)

        if not candidates:
            return None

        examples: list[str] = []
        example_candidate_indices: list[int] = []

        for candidate_index, candidate in enumerate(candidates):
            for example in candidate.examples:
                examples.append(example)
                example_candidate_indices.append(candidate_index)

        target_embeddings = self.encoder.encode(examples)
        scores = self.encoder.cosine_scores(
            query_embedding=message_embedding,
            candidate_embeddings=target_embeddings,
        )

        candidate_scores = [float("-inf")] * len(candidates)
        for score, candidate_index in zip(scores.detach().cpu().tolist(), example_candidate_indices):
            candidate_scores[candidate_index] = max(candidate_scores[candidate_index], float(score))

        ranked = sorted(
            enumerate(candidate_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        top_index, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")

        if self.debug:
            named_scores = [
                (self._target_debug_label(candidates[index].target), score)
                for index, score in ranked
            ]
            self._print_ranked_scores("Target scores", named_scores)

        if top_score < self.target_threshold:
            return None

        if top_score - second_score < self.target_margin:
            return None

        return TargetMatch(
            target=candidates[top_index].target,
            score=top_score,
            second_score=second_score,
        )

    def build_target_candidates(self, world: WorldView) -> list[TargetCandidate]:
        candidates = [
            TargetCandidate(
                target=TargetRef.player(),
                examples=(
                    "me",
                    "myself",
                    "the player",
                    "the human player",
                    "where I am",
                    "my position",
                    "here",
                    "near me",
                    "come here",
                    "follow me",
                    "give me space",
                    "move away from me",
                    "stay away from me",
                ),
            )
        ]

        for object_view in world.objects:
            candidates.append(
                TargetCandidate(
                    target=TargetRef.object(
                        object_id=object_view.object_id,
                        label=object_view.name,
                    ),
                    examples=self._object_target_examples(object_view),
                )
            )

        return candidates

    def _object_target_examples(self, object_view: ObjectView) -> tuple[str, ...]:
        tag_text = " ".join(sorted(object_view.tags))
        color_kind = f"{object_view.color_name} {object_view.kind}".strip()

        examples = [
            object_view.name,
            color_kind,
            f"the {color_kind}",
            f"the {object_view.color_name} object",
            f"the {object_view.color_name} thing",
            f"the {object_view.kind}",
            f"the object {object_view.direction_label} of you",
            f"the {color_kind} {object_view.direction_label} of you",
            f"the nearby {color_kind}",
        ]

        if tag_text:
            examples.append(tag_text)
            examples.append(f"the {tag_text} object")

        return tuple(self._unique_non_empty(examples))

    def _spec_for_action(self, action: NPCAction) -> ActionSpec:
        for spec in self.action_specs:
            if spec.action == action:
                return spec

        raise ValueError(f"No semantic action spec for action: {action}")

    def _target_debug_label(self, target: TargetRef) -> str:
        if target.kind == "object":
            return f"object:{target.label}"

        return target.kind

    def _print_ranked_scores(
        self,
        title: str,
        ranked_scores: list[tuple[str, float]] | list[tuple[NPCAction, float]],
    ) -> None:
        print(title)
        for label, score in ranked_scores:
            print(f"  {label}: {score:.3f}")

    def _unique_non_empty(self, values: Iterable[str]) -> list[str]:
        seen = set()
        unique = []

        for value in values:
            normalized = " ".join(value.split())

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(normalized)

        return unique


class NPCBrain:
    def __init__(
        self,
        encoder_model_name: str = ENCODER_MODEL_NAME,
        decoder_model_name: str = DECODER_MODEL_NAME,
        encoder_device: str | None = None,
        decoder_device: str | None = None,
        use_4bit_decoder: bool = True,
        debug: bool = False,
    ):
        self.debug = debug
        self.semantic_encoder = NPCSemanticEncoder(
            model_name=encoder_model_name,
            device=encoder_device,
        )
        self.command_mapper = SemanticCommandMapper(
            encoder=self.semantic_encoder,
            debug=debug,
        )
        self.speech_decoder = NPCSpeechDecoder(
            model_name=decoder_model_name,
            device=decoder_device,
            use_4bit=use_4bit_decoder,
        )

    def decide(
        self,
        player_message: str,
        world: WorldView,
    ) -> NPCCommand:
        command = self.command_mapper.map_message_to_command(
            player_message=player_message,
            world=world,
        )

        if command.action != "respond_text":
            return command

        if command.text:
            return command

        return NPCCommand.respond_text(
            self.speech_decoder.generate(
                context_text=self._world_context_text(world),
                latest_player_text=player_message,
            )
        )

    def _world_context_text(self, world: WorldView) -> str:
        if not world.objects:
            return "The NPC sees no nearby objects."

        visible_objects = [self._object_context_text(object_view) for object_view in world.objects[:5]]
        return "The NPC can see: " + "; ".join(visible_objects) + "."

    def _object_context_text(self, object_view: ObjectView) -> str:
        distance_label = self._distance_label(object_view.distance_to_npc)
        color_kind = f"{object_view.color_name} {object_view.kind}".strip()
        return f"a {distance_label} {color_kind} named {object_view.name} {object_view.direction_label} of the NPC"

    def _distance_label(self, distance: float) -> str:
        if distance < 3.0:
            return "nearby"
        if distance < 8.0:
            return "midrange"
        return "faraway"
