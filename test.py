import numpy as np

def identify_exposed_voxels(voxel_world):
    # Pad the voxel world with zeros (air) on all sides
    padded_world = np.pad(voxel_world, pad_width=1, mode='constant', constant_values=0)
    
    # Create shifted versions of the world for all six directions
    shifts = {
        'left':  (0, -1, 0),
        'right': (0, 1, 0),
        'down':  (-1, 0, 0),
        'up':    (1, 0, 0),
        'back':  (0, 0, -1),
        'front': (0, 0, 1),
    }
    exposed_faces = np.zeros_like(voxel_world, dtype=bool)
    
    for direction, (dx, dy, dz) in shifts.items():
        shifted_world = np.roll(padded_world, shift=(dx, dy, dz), axis=(0, 1, 2))
        # Expose face if there's air next to it (voxel value of 0 in the shifted world)
        exposed_faces |= ((shifted_world[1:-1, 1:-1, 1:-1] == 0) & (voxel_world > 0))
    
    return exposed_faces

def check_surrounding_air_vectorized(voxel_world, x, y, z):
    """
    Vectorized version to check each of the six directions around a point (x, y, z) in the voxel world
    for air (assumed to be represented by 0).
    """
    # Pad the voxel world with 1 (solid) to handle edge cases without manual boundary checks
    padded_world = np.pad(voxel_world, pad_width=1, mode='constant', constant_values=0)

    # Adjust coordinates due to padding
    x, y, z = x + 1, y + 1, z + 1

    # Check the six directions around the point for air, vectorized
    left = padded_world[x - 1, y, z] == 0
    right = padded_world[x + 1, y, z] == 0
    down = padded_world[x, y - 1, z] == 0
    up = padded_world[x, y + 1, z] == 0
    front = padded_world[x, y, z - 1] == 0
    back = padded_world[x, y, z + 1] == 0

    faces = ["left", "right", "down", "up", "front", "back"]
    exposed = [back, front, down, up, right, left]
    
    return [face for face, exp in zip(faces, exposed) if exp]

voxel_world = np.zeros((3, 3, 3), dtype=int)
voxel_world[1, 1, 1] = 1

voxels = identify_exposed_voxels(voxel_world)

print(voxels)



print(check_surrounding_air_vectorized(voxel_world, 1, 1, 1))