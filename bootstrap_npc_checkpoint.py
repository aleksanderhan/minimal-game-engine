from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from npc_model import ACTION_NAMES, NPCModel
from npc_agent import OBS_DIM


def action_index(name: str) -> int:
    return ACTION_NAMES.index(name)


def sample_observations(batch_size: int, device: torch.device) -> torch.Tensor:
    npc_pos = torch.empty(batch_size, 3, device=device).uniform_(-5.0, 5.0)
    npc_pos[:, 2] = torch.empty(batch_size, device=device).uniform_(1.0, 4.0)

    npc_velocity = torch.empty(batch_size, 3, device=device).uniform_(-0.5, 0.5)

    player_pos = torch.empty(batch_size, 3, device=device).uniform_(-5.0, 5.0)
    player_pos[:, 2] = torch.empty(batch_size, device=device).uniform_(1.0, 4.0)

    to_player = player_pos - npc_pos
    flat_to_player = to_player.clone()
    flat_to_player[:, 2] = 0.0

    distance = torch.linalg.norm(flat_to_player[:, :2], dim=-1).clamp_min(1e-6)
    direction = flat_to_player[:, :2] / distance.unsqueeze(-1)

    obs = torch.cat(
        [
            npc_pos,
            npc_velocity,
            player_pos,
            direction,
            distance.unsqueeze(-1),
        ],
        dim=-1,
    )

    return obs


def make_prompts(obs: torch.Tensor) -> list[str]:
    obs_cpu = obs.detach().cpu().numpy()
    prompts: list[str] = []

    for row in obs_cpu:
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
        ) = row.tolist()

        prompts.append(
            f"""
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
        )

    return prompts


def scripted_labels(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = obs.shape[0]
    device = obs.device

    distance = obs[:, 11]

    action_targets = torch.full(
        (batch_size,),
        action_index("idle"),
        dtype=torch.long,
        device=device,
    )

    control_targets = torch.zeros(batch_size, 3, device=device)
    speak_targets = torch.zeros(batch_size, 1, device=device)
    value_targets = torch.zeros(batch_size, 1, device=device)

    too_far = distance > 4.0
    too_close = distance < 1.5
    conversational = (distance >= 1.5) & (distance <= 4.0)

    action_targets[too_far] = action_index("approach_player")
    control_targets[too_far, 0] = 0.8

    action_targets[too_close] = action_index("back_away")
    control_targets[too_close, 0] = 0.7

    action_targets[conversational] = action_index("speak")
    control_targets[conversational, 0] = 0.0
    speak_targets[conversational, 0] = 1.0

    value_targets[:, 0] = torch.exp(-torch.abs(distance - 2.5))

    return action_targets, control_targets, speak_targets, value_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    parser.add_argument("--output", default="npc_policy.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = NPCModel(
        model_name=args.model_name,
        obs_dim=OBS_DIM,
        num_actions=len(ACTION_NAMES),
        device=args.device,
        use_4bit=True,
    )

    for param in model.llm.parameters():
        param.requires_grad = False

    trainable_params = [
        *model.obs_encoder.parameters(),
        *model.fusion.parameters(),
        *model.action_head.parameters(),
        *model.control_head.parameters(),
        *model.speak_head.parameters(),
        *model.value_head.parameters(),
    ]

    optimizer = AdamW(trainable_params, lr=args.lr)

    model.train()
    model.llm.eval()

    for step in range(args.steps):
        obs = sample_observations(args.batch_size, device)
        prompts = make_prompts(obs)

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        action_targets, control_targets, speak_targets, value_targets = scripted_labels(obs)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            obs=obs,
        )

        action_loss = F.cross_entropy(outputs["action_logits"], action_targets)
        control_loss = F.mse_loss(outputs["control"], control_targets)
        speak_loss = F.binary_cross_entropy_with_logits(outputs["speak_logits"], speak_targets)
        value_loss = F.mse_loss(outputs["value"], value_targets)

        loss = (
            action_loss
            + control_loss
            + speak_loss
            + 0.25 * value_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(
                {
                    "step": step,
                    "loss": round(float(loss.item()), 4),
                    "action_loss": round(float(action_loss.item()), 4),
                    "control_loss": round(float(control_loss.item()), 4),
                    "speak_loss": round(float(speak_loss.item()), 4),
                    "value_loss": round(float(value_loss.item()), 4),
                }
            )

    npc_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("llm.")
    }

    torch.save(
        {
            "npc_model": npc_state_dict,
            "action_names": ACTION_NAMES,
            "obs_dim": OBS_DIM,
            "model_name": args.model_name,
            "use_4bit": "true",
            "bnb_4bit_quant_type": "nf4",
        },
        args.output,
    )

    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()