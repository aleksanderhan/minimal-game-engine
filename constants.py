import numpy as np

offset_arrays = {
    "front": np.array([(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
    "back": np.array([(1, -1, -1), (1, -1, 1), (-1, -1, 1), (-1, -1, -1)]),
    "right": np.array([(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
    "left": np.array([(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
    "up": np.array([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    "down": np.array([(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)]),
}


uv_maps = {
    0: {
        "front": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
        "back": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
        "right": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
        "left": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
        "up": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
        "down": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
    }, 
    1: {
        "front": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
        "back": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
        "right": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
        "left": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
        "up": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
        "down": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
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
    (0, 0, -1): (0, 1, 1, 1)  # cyan -down
}