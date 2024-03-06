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


voxel_world = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])


voxels = identify_exposed_voxels(voxel_world)

print(voxels)