import numpy as np
from enum import Enum

class VoxelType(Enum):
    AIR = 0
    STONE = 1
    GRASS = 2

voxel_type_map = {
    0: VoxelType.AIR,
    1: VoxelType.STONE,
    2: VoxelType.GRASS,
}

offset_arrays = {
    "front": np.array([(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)]),
    "back": np.array([(0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5)]),
    "right": np.array([(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
    "left": np.array([(-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5)]),
    "up": np.array([(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
    "down": np.array([(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]),
}


uv_maps = {
    VoxelType.AIR: {
        "front": [(0, 0), (0, 0), (0, 0), (0, 0)],
        "back": [(0, 0), (0, 0), (0, 0), (0, 0)],
        "right": [(0, 0), (0, 0), (0, 0), (0, 0)],
        "left": [(0, 0), (0, 0), (0, 0), (0, 0)],
        "up": [(0, 0), (0, 0), (0, 0), (0, 0)],
        "down": [(0, 0), (0, 0), (0, 0), (0, 0)],
    },
    VoxelType.STONE: {
        "front": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
        "back": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
        "right": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
        "left": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
        "up": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
        "down": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
    }, 
    VoxelType.GRASS: {
        "front": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
        "back": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
        "right": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
        "left": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
        "up": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
        "down": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
    },
}

# Define offsets for each face (adjust based on your coordinate system)
normals = {
    'front':  ( 0,  1,  0),
    'back':   (0,  -1,  0),
    'right':  ( 1,  0,  0),
    'left':   ( -1, 0,  0),
    'up':   ( 0,  0, 1),
    'down':     ( 0,  0,  -1),
}

color_normal_map = {
    (0, 1, 0): (1, 0, 0, 1), # red - front
    (0, -1, 0): (0, 0, 1, 1), # blue  - back
    (1, 0, 0): (1, 0, 1, 1), # magenta - right
    (-1, 0, 0): (1, 1, 0, 1), # yellow -left
    (0, 0, 1): (0, 1, 0, 1), # green - up
    (0, 0, -1): (0, 1, 1, 1),  # cyan -down
}

normal_face_map = {
    (0, 1, 0): "front",
    (0, -1, 0): "back",
    (1, 0, 0): "right",
    (-1, 0, 0): "left",
    (0, 0, 1): "up",
    (0, 0, -1): "down",
}

material_properties = {
    VoxelType.AIR: {
        "color": (0, 0, 0, 0),
        "mass": 0,
        "friction": 0,
        "coupling_strength": 0
    },
    VoxelType.STONE: {
        "color": (0.47, 0.51, 0.51, 0.5),
        "mass": 10,
        "friction": 100,
        "coupling_strength": 1 
    },
    VoxelType.GRASS: {
        "color": (0.26, 0.87, 0.31, 0.5),
        "mass": 2,
        "friction": 2,
        "coupling_strength": 0.5
    },
}