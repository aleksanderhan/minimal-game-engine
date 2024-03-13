import numpy as np
from constants import VoxelType, voxel_type_map

class VoxelWorld:

    def __init__(self, world_array, voxel_size):
        self.voxel_size = voxel_size
        self.world_array = world_array
        
    def get_voxel_type(self, ix, iy, iz) -> VoxelType: 
        # Assuming world_array is centered around (0, 0, 0) at initialization
        # and offset is half the size of the current array dimensions.
        i, j, k = VoxelWorld.voxel_grid_coordinates_to_index(ix, iy, iz, self.world_array.shape[0])
        voxel_type_int = self.world_array[i, j, k]
        return voxel_type_map[voxel_type_int]

    def set_voxel(self, ix, iy, iz, voxel_type: VoxelType):
        i, j, k = VoxelWorld.voxel_grid_coordinates_to_index(ix, iy, iz, self.world_array.shape[0])
        self.world_array[i, j, k] = voxel_type.value
    
    @staticmethod
    def voxel_grid_coordinates_to_index(ix, iy, iz, n):
        print("x, y, z, n, voxel_size", ix, iy, iz, n)
        """
        Convert world coordinates (ix, iy, iz) to array indices (i, j, k).
        
        Parameters:
        - ix, iy, iz: World grid coordinates.
        - n: Size of the voxel world in each dimension.
        
        Returns:
        - i, j, k: Corresponding array indices.
        """
        offset = (n - 1) // 2
        i = ix + offset
        j = iy + offset
        k = iz  # No change needed for z as it cannot be negative.
        return i, j, k

    @staticmethod
    def index_to_voxel_grid_coordinates(i, j, k, n):
        """
        Convert array indices (i, j, k) back to world grid coordinates (ix, iy, iz).
        
        Parameters:
        - i, j, k: Array indices.
        - n: Size of the voxel world in each dimension.
        
        Returns:
        - ix, iy, iz: Corresponding world coordinates.
        """
        offset = (n - 1) // 2
        ix = i - offset
        iy = j - offset
        iz = k - 1 # No change needed for z as it cannot be negative.
        
        return ix, iy, iz

class WorldChunk:

    def __init__(self):
        pass