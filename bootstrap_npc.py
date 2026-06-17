from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


ACTION_LABELS = (
    "move_towards",
    "move_away",
    "stop",
    "respond_text",
)

ActionLabel = Literal[
    "move_towards",
    "move_away",
    "stop",
    "respond_text",
]

ACTION_TO_INDEX = {
    label: index
    for index, label in enumerate(ACTION_LABELS)
}


ENCODER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


COLORS = (
    "red",
    "green",
    "blue",
    "yellow",
    "white",
    "black",
    "gray",
    "orange",
    "purple",
)

KINDS = (
    "ball",
    "cube",
    "box",
    "voxel",
    "sphere",
    "object",
)

OBJECT_PHRASES = (
    "red ball",
    "green ball",
    "blue ball",
    "yellow cube",
    "gray voxel",
    "orange sphere",
    "purple object",
    "nearest ball",
    "closest cube",
    "the object",
    "that object",
    "the thing",
)

PLAYER_TARGET_PHRASES = (
    "me",
    "the player",
    "here",
    "my position",
    "where I am",
)


MOVE_TOWARDS_TEMPLATES = (
    "come here",
    "come to me",
    "move to me",
    "walk to me",
    "follow me",
    "get closer to me",
    "approach me",
    "move towards me",
    "come over here",
    "stand near me",
    "go to the {target}",
    "move to the {target}",
    "walk to the {target}",
    "approach the {target}",
    "get closer to the {target}",
    "stand near the {target}",
    "head over to the {target}",
    "go towards the {target}",
    "move towards the {target}",
)

MOVE_AWAY_TEMPLATES = (
    "move away",
    "go away",
    "back away",
    "back up",
    "give me space",
    "step away from me",
    "move away from me",
    "keep your distance from me",
    "do not stand so close",
    "back away from the {target}",
    "move away from the {target}",
    "step away from the {target}",
    "keep away from the {target}",
    "get away from the {target}",
    "avoid the {target}",
)

STOP_TEMPLATES = (
    "stop",
    "freeze",
    "hold still",
    "stop moving",
    "stay there",
    "stay where you are",
    "wait",
    "wait there",
    "cancel that",
    "never mind",
    "do not move",
    "stand still",
    "stop following me",
    "stop going there",
)

RESPOND_TEXT_TEMPLATES = (
    "hello",
    "hey",
    "hi",
    "what do you see",
    "what is nearby",
    "describe the area",
    "what objects are around you",
    "can you see the red ball",
    "where is the nearest ball",
    "tell me what you see",
    "say something",
    "are you awake",
    "do you understand me",
    "what are you doing",
    "how are you",
    "which objects are close to you",
)


@dataclass(frozen=True)
class TrainingSample:
    text: str
    label: ActionLabel


class NPCActionDataset(Dataset):
    def __init__(self, samples: list[TrainingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        return self.samples[index]


class NPCActionClassifierModel(nn.Module):
    def __init__(
        self,
        encoder_model_name: str,
        hidden_dim: int = 512,
        dropout: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()

        self.encoder_model_name = encoder_model_name
        self.device_name = device
        self.runtime_device = torch.device(device)

        self.encoder = AutoModel.from_pretrained(
            encoder_model_name,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(self.runtime_device)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        encoder_dim = int(self.encoder.config.hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ACTION_LABELS)),
        ).to(self.runtime_device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
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
            )

        pooled = self.mean_pool(
            hidden=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

        return self.classifier(pooled.float())

    def mean_pool(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        return summed / count


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_target_phrase() -> str:
    if random.random() < 0.35:
        return random.choice(PLAYER_TARGET_PHRASES)

    return random.choice(OBJECT_PHRASES)


def maybe_politeness(text: str) -> str:
    roll = random.random()

    if roll < 0.15:
        return f"please {text}"

    if roll < 0.30:
        return f"can you {text}"

    if roll < 0.42:
        return f"could you {text} please"

    if roll < 0.52:
        return f"npc, {text}"

    return text


def maybe_noise(text: str) -> str:
    roll = random.random()

    if roll < 0.15:
        return f"hey, {text}"

    if roll < 0.25:
        return f"{text} now"

    if roll < 0.35:
        return f"{text} for a moment"

    if roll < 0.42:
        return f"{text}, thanks"

    return text


def render_template(template: str) -> str:
    if "{target}" in template:
        return template.format(target=random_target_phrase())

    return template


def make_world_context() -> str:
    objects = random.sample(
        list(OBJECT_PHRASES),
        k=random.randint(2, 6),
    )

    lines = ["Visible objects:"]

    for index, object_phrase in enumerate(objects, start=1):
        distance = random.uniform(1.0, 12.0)
        direction = random.choice(("front", "behind", "left", "right", "nearby"))

        lines.append(
            f"- object_{index}: {object_phrase}, {distance:.1f}m, {direction}"
        )

    return "\n".join(lines)


def make_policy_text(player_message: str, include_world_context: bool) -> str:
    sections = [
        "You control an NPC in a physics sandbox.",
        "Choose exactly one action.",
        "",
        "Available actions:",
        "- move_towards",
        "- move_away",
        "- stop",
        "- respond_text",
        "",
        f'Player message: "{player_message}"',
    ]

    if include_world_context:
        sections.extend(
            [
                "",
                make_world_context(),
            ]
        )

    return "\n".join(sections)


def make_samples_for_action(
    action: ActionLabel,
    count: int,
    include_world_context_probability: float,
) -> list[TrainingSample]:
    if action == "move_towards":
        templates = MOVE_TOWARDS_TEMPLATES
    elif action == "move_away":
        templates = MOVE_AWAY_TEMPLATES
    elif action == "stop":
        templates = STOP_TEMPLATES
    elif action == "respond_text":
        templates = RESPOND_TEXT_TEMPLATES
    else:
        raise ValueError(f"Unhandled action: {action}")

    samples = []

    for _ in range(count):
        template = random.choice(templates)
        message = render_template(template)
        message = maybe_politeness(message)
        message = maybe_noise(message)

        include_world_context = random.random() < include_world_context_probability

        text = make_policy_text(
            player_message=message,
            include_world_context=include_world_context,
        )

        samples.append(
            TrainingSample(
                text=text,
                label=action,
            )
        )

    return samples


def make_dataset(
    samples_per_action: int,
    include_world_context_probability: float,
) -> list[TrainingSample]:
    samples: list[TrainingSample] = []

    for action in ACTION_LABELS:
        samples.extend(
            make_samples_for_action(
                action=action,
                count=samples_per_action,
                include_world_context_probability=include_world_context_probability,
            )
        )

    random.shuffle(samples)

    return samples


def split_dataset(
    samples: list[TrainingSample],
    validation_fraction: float,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    random.shuffle(samples)

    validation_count = int(len(samples) * validation_fraction)

    validation_samples = samples[:validation_count]
    training_samples = samples[validation_count:]

    return training_samples, validation_samples


def make_collate_fn(
    tokenizer,
    max_length: int,
):
    def collate(samples: list[TrainingSample]) -> dict[str, torch.Tensor]:
        texts = [sample.text for sample in samples]

        labels = torch.tensor(
            [ACTION_TO_INDEX[sample.label] for sample in samples],
            dtype=torch.long,
        )

        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    return collate


def train_one_epoch(
    model: NPCActionClassifierModel,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: str,
    grad_clip_norm: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in data_loader:
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.classifier.parameters(),
                grad_clip_norm,
            )

        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: NPCActionClassifierModel,
    data_loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in data_loader:
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((predictions == labels).sum().detach().cpu())
        total_count += batch_size

    average_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)

    return average_loss, accuracy


def print_label_distribution(
    name: str,
    samples: list[TrainingSample],
) -> None:
    counts = Counter(sample.label for sample in samples)

    print(f"{name} label distribution:")

    for label in ACTION_LABELS:
        print(f"  {label}: {counts[label]}")


def save_checkpoint(
    path: str,
    model: NPCActionClassifierModel,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "architecture": "npc_action_classifier_v1",
        "action_labels": ACTION_LABELS,
        "encoder_model_name": args.encoder_model_name,
        "max_length": args.max_length,
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "args": vars(args),
    }

    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the simplified NPC action classifier.",
    )

    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="npc_action_classifier.pt",
    )

    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default=ENCODER_MODEL_NAME,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--samples-per-action",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--include-world-context-probability",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.0e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)

    samples = make_dataset(
        samples_per_action=args.samples_per_action,
        include_world_context_probability=args.include_world_context_probability,
    )

    training_samples, validation_samples = split_dataset(
        samples=samples,
        validation_fraction=args.validation_fraction,
    )

    print_label_distribution("training", training_samples)
    print_label_distribution("validation", validation_samples)

    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    training_loader = DataLoader(
        NPCActionDataset(training_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    validation_loader = DataLoader(
        NPCActionDataset(validation_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = NPCActionClassifierModel(
        encoder_model_name=args.encoder_model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=args.device,
    )

    optimizer = AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        training_loss = train_one_epoch(
            model=model,
            data_loader=training_loader,
            optimizer=optimizer,
            device=args.device,
            grad_clip_norm=args.grad_clip_norm,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            device=args.device,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={training_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            save_checkpoint(
                path=args.checkpoint_out,
                model=model,
                args=args,
            )

            print(f"saved checkpoint: {args.checkpoint_out}")

    print(f"best validation accuracy: {best_validation_accuracy:.4f}")


if __name__ == "__main__":
    main()