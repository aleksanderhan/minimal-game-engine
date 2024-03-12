import numpy as np
from constants import VoxelType, voxel_type_map

class VoxelWorld:

    def __init__(self, world_array):
        self.world_array = world_array
        self.offset = (np.array(world_array.shape) - 1) // 2
        
    def get_voxel_type(self, x, y, z) -> VoxelType: 
        # Assuming world_array is centered around (0, 0, 0) at initialization
        # and offset is half the size of the current array dimensions.
        ix, iy, iz = x + self.offset[0], y + self.offset[1], z + self.offset[2] - 1
        voxel_type_int = self.world_array[ix, iy, iz]
        return voxel_type_map[voxel_type_int]

    def set_voxel(self, x, y, z, voxel_type: VoxelType):
        ix, iy, iz = x + self.offset[0], y + self.offset[1], z + self.offset[2] - 1        
        self.world_array[ix, iy, iz] = voxel_type.value

    def get_world_array(self):
        return self.world_array
    
    def shape(self):
        return self.world_array.shape
    
    @staticmethod
    def world_to_index(x, y, z, n):
        """
        Convert world coordinates (x, y, z) to array indices (i, j, k).
        
        Parameters:
        - x, y, z: World coordinates.
        - n: Size of the voxel world in each dimension.
        
        Returns:
        - i, j, k: Corresponding array indices.
        """
        offset = (n - 1) // 2
        i = x + offset
        j = y + offset
        k = z  # No change needed for z as it cannot be negative.
        
        return i, j, k

    @staticmethod
    def index_to_world(i, j, k, n):
        """
        Convert array indices (i, j, k) back to world coordinates (x, y, z).
        
        Parameters:
        - i, j, k: Array indices.
        - n: Size of the voxel world in each dimension.
        
        Returns:
        - x, y, z: Corresponding world coordinates.
        """
        offset = (n - 1) // 2
        x = i - offset
        y = j - offset
        z = k - 1 # No change needed for z as it cannot be negative.
        
        return x, y, z

class WorldChunk:

    def __init__(self):
        pass