from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ACTION_NAMES = [
    "idle",
    "approach_player",
    "back_away",
    "strafe_left",
    "strafe_right",
    "jump",
    "speak",
]


@dataclass(frozen=True)
class NPCPolicyOutput:
    action_name: str
    speed: float
    strafe: float
    jump: float
    should_speak: bool
    utterance: str


class NPCModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        obs_dim: int,
        num_actions: int,
        device: str,
        use_4bit: bool,
    ):
        super().__init__()

        self.device_name = device
        self.runtime_device = torch.device(device)

        quantization_config = None
        device_map = None

        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": 0} if device.startswith("cuda") else {"": device}

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            output_hidden_states=True,
        )

        hidden_dim = self.llm.config.text_config.hidden_size

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.runtime_device)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.runtime_device)

        self.action_head = nn.Linear(hidden_dim, num_actions).to(self.runtime_device)

        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        ).to(self.runtime_device)

        self.speak_head = nn.Linear(hidden_dim, 1).to(self.runtime_device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.runtime_device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        obs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_ids = input_ids.to(self.runtime_device)
        attention_mask = attention_mask.to(self.runtime_device)
        obs = obs.to(self.runtime_device)

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]

        last_valid_indices = (
            attention_mask.shape[1]
            - 1
            - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
        )

        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )

        text_hidden = last_hidden[batch_indices, last_valid_indices, :].float()
        obs_hidden = self.obs_encoder(obs.float())

        if text_hidden.shape[-1] != obs_hidden.shape[-1]:
            raise RuntimeError(
                f"text_hidden dim {text_hidden.shape[-1]} does not match "
                f"obs_hidden dim {obs_hidden.shape[-1]}"
            )

        fused_hidden = self.fusion(
            torch.cat(
                [text_hidden, obs_hidden],
                dim=-1,
            )
        )

        return {
            "language_logits": outputs.logits,
            "action_logits": self.action_head(fused_hidden),
            "control": self.control_head(fused_hidden),
            "speak_logits": self.speak_head(fused_hidden),
            "value": self.value_head(fused_hidden),
        }


class NPCBrain:
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        obs_dim: int,
        device: str,
    ):
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = NPCModel(
            model_name=model_name,
            obs_dim=obs_dim,
            num_actions=len(ACTION_NAMES),
            device=device,
            use_4bit=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        self.model.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray, prompt: str) -> NPCPolicyOutput:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        obs = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            obs=obs,
        )

        action_index = int(torch.argmax(output["action_logits"], dim=-1).item())
        action_name = ACTION_NAMES[action_index]

        control = output["control"][0].detach().cpu().numpy()
        speed = float((control[0] + 1.0) * 0.5)
        strafe = float(control[1])
        jump = float(max(control[2], 0.0))

        should_speak = bool(torch.sigmoid(output["speak_logits"])[0, 0].item() > 0.5)
        utterance = ""

        if should_speak:
            utterance = self._generate_dialogue(prompt)

        return NPCPolicyOutput(
            action_name=action_name,
            speed=speed,
            strafe=strafe,
            jump=jump,
            should_speak=should_speak,
            utterance=utterance,
        )

    @torch.no_grad()
    def _generate_dialogue(self, prompt: str) -> str:
        dialogue_prompt = prompt + "\nNPC:"
        encoded = self.tokenizer(
            dialogue_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        generated = self.model.llm.generate(
            **encoded,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text[len(dialogue_prompt):].strip().split("\n")[0]