from enum import Enum

class VoxelType(Enum):
    PLACEHOLDER_BLOCK = -1
    AIR = 0
    STONE = 1
    GRASS = 2

voxel_type_map = {
    255: VoxelType.PLACEHOLDER_BLOCK,
    0: VoxelType.AIR,
    1: VoxelType.STONE,
    2: VoxelType.GRASS,
}

color_normal_map = {
    (0, 1, 0): (1, 0, 0, 1), # red - front
    (0, -1, 0): (0, 0, 1, 1), # blue  - back
    (1, 0, 0): (1, 0, 1, 1), # magenta - right
    (-1, 0, 0): (1, 1, 0, 1), # yellow -left
    (0, 0, 1): (0, 1, 0, 1), # green - up
    (0, 0, -1): (0, 1, 1, 1),  # cyan -down
}

material_properties = {
    VoxelType.PLACEHOLDER_BLOCK: {
        "color": (0.47, 0.51, 0.51, 0.8),
        "mass": 0,
        "friction": 0,
        "coupling_strength": 0,
    },
    VoxelType.AIR: {
        "color": (0, 0, 0, 0),
        "mass": 0,
        "friction": 0,
        "coupling_strength": 0,
    },
    VoxelType.STONE: {
        "color": (0.47, 0.51, 0.51, 0.8),
        "mass": 50,
        "friction": 100,
        "coupling_strength": 1,
    },
    VoxelType.GRASS: {
        "color": (0.26, 0.87, 0.31, 0.8),
        "mass": 10,
        "friction": 10,
        "coupling_strength": 0.5,
    },
}
