import numpy as np
from functools import lru_cache
from typing import Any

from constants import VoxelType, material_properties


# Toggle generator. Returns a or b alternatingly on next()
def toggle(a, b, yield_a=True):
    while True:
        (yield a) if yield_a else (yield b)
        yield_a = not yield_a

# We cant use enums in jit functions, so we have to compute the voxel_type_color map before and send it to creat_mesh
@lru_cache
def create_voxel_type_value_color_list() -> list[tuple[float, float, float, float]]:
    return [material_properties[voxel_type]["color"] for voxel_type in VoxelType]