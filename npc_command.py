from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from panda3d.core import Point3


NPCAction = Literal[
    "move_towards",
    "move_away",
    "stop",
    "respond_text",
]

TargetKind = Literal[
    "player",
    "object",
    "position",
    "none",
]


@dataclass(frozen=True)
class TargetRef:
    kind: TargetKind
    label: str = ""
    object_id: str | None = None
    position: Point3 | None = None

    @classmethod
    def player(cls) -> "TargetRef":
        return cls(kind="player", label="player")

    @classmethod
    def object(cls, object_id: str, label: str) -> "TargetRef":
        return cls(kind="object", object_id=object_id, label=label)

    @classmethod
    def position(cls, position: Point3, label: str = "position") -> "TargetRef":
        return cls(kind="position", position=position, label=label)

    @classmethod
    def none(cls) -> "TargetRef":
        return cls(kind="none", label="")


@dataclass(frozen=True)
class NPCCommand:
    action: NPCAction
    target: TargetRef
    text: str = ""

    @classmethod
    def move_towards(cls, target: TargetRef) -> "NPCCommand":
        return cls(action="move_towards", target=target)

    @classmethod
    def move_away(cls, target: TargetRef) -> "NPCCommand":
        return cls(action="move_away", target=target)

    @classmethod
    def stop(cls) -> "NPCCommand":
        return cls(action="stop", target=TargetRef.none())

    @classmethod
    def respond_text(cls, text: str) -> "NPCCommand":
        return cls(action="respond_text", target=TargetRef.none(), text=text)