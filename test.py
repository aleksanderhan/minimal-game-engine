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

    faces = ["left", "right", "down", "up", "back", "front"]
    exposed = [left, right, down, up, back, front]
    
    return [face for face, exp in zip(faces, exposed) if exp]

voxel_world1 = np.zeros((3,3,3), dtype=int)
voxel_world1[0, 0, 0] = 1
voxel_world1[0, 1, 0] = 1

voxel_world2 = np.zeros((3,3,3), dtype=int)
voxel_world2[1, 1, 0] = 1
voxel_world2[1, 1, 1] = 1


voxels1 = identify_exposed_voxels(voxel_world1)
voxels2 = identify_exposed_voxels(voxel_world2)

print("identify_exposed_voxels1:", voxels1)
print("identify_exposed_voxels2:", voxels2)


print("check_surrounding_air_vectorized1:", check_surrounding_air_vectorized(voxel_world1, 1, 1, 1))
print("check_surrounding_air_vectorized2:", check_surrounding_air_vectorized(voxel_world2, 1, 1, 1))


# Defining a mock class to simulate the VoxelWorld behavior for unit testing
class MockVoxelWorld:
    def __init__(self, array):
        self.array = array
    
    def shape(self):
        return self.array.shape
    
    def get_voxel(self, x, y, z):
        return self.array[x, y, z]

# Re-defining the check_surrounding_air function to use the MockVoxelWorld
def check_surrounding_air(voxel_world: MockVoxelWorld, x, y, z):
    max_x, max_y, max_z = voxel_world.shape()[0] - 1, voxel_world.shape()[1] - 1, voxel_world.shape()[2] - 1
    exposed_faces = []
    
    if x == max_x or voxel_world.get_voxel(x + 1, y, z) == 0: exposed_faces.append("right")
    if x == 0 or voxel_world.get_voxel(x - 1, y, z) == 0: exposed_faces.append("left")
    if y == max_y or voxel_world.get_voxel(x, y + 1, z) == 0: exposed_faces.append("front")
    if y == 0 or voxel_world.get_voxel(x, y - 1, z) == 0: exposed_faces.append("back")
    if z == max_z or voxel_world.get_voxel(x, y, z + 1) == 0: exposed_faces.append("up")
    if z == 0 or voxel_world.get_voxel(x, y, z - 1) == 0: exposed_faces.append("down")
    
    return exposed_faces

# Writing a unit test
def test_check_surrounding_air():
    import numpy as np
    voxel_array = np.array([[[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 1, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]])
    voxel_world = MockVoxelWorld(voxel_array)
    result = check_surrounding_air(voxel_world, 3, 3, 2)
    expected = ["right", "left", "front", "back", "up", "down"]
    print("result", result)
    assert set(result) == set(expected), f"Expected {expected}, but got {result}"

# Running the unit test
test_check_surrounding_air()