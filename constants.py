from enum import Enum

class VoxelType(Enum):
    PLACEHOLDER_BLOCK = -1
    AIR = 0
    STONE = 1
    GRASS = 2

voxel_type_map = {voxel_type.value: voxel_type for voxel_type in VoxelType}

material_properties = {
    VoxelType.PLACEHOLDER_BLOCK: {
        "color": (0.5, 0.5, 0.8, 0.5),
        "mass": 0,
        "friction": 0,
        "coupling_strength": 0,
        "restitution": 0,
        "linear_damping": 0,
        "angular_damping": 0,
    },
    VoxelType.AIR: {
        "color": (0.0, 0.0, 0.0, 0.0),
        "mass": 0,
        "friction": 0,
        "coupling_strength": 0,
        "restitution": 0,
        "linear_damping": 0,
        "angular_damping": 0,
    },
    VoxelType.STONE: {
        "color": (0.47, 0.51, 0.51, 0.8),
        "mass": 50,
        "friction": 10.0,
        "coupling_strength": 1.0,
        "restitution": 1.0,
        "linear_damping": 0.2,
        "angular_damping": 0.2,
    },
    VoxelType.GRASS: {
        "color": (0.26, 0.87, 0.31, 0.8),
        "mass": 10,
        "friction": 5.0,
        "coupling_strength": 0.5,
        "restitution": 0.5,
        "linear_damping": 0.01,
        "angular_damping": 0.01,
    },
}
