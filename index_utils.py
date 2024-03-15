
class IndexTools:

    @staticmethod
    def voxel_grid_coordinate_to_index(ix: int, iy: int, iz: int, n: int) -> tuple[str]:
        """
        Convert world grid coordinate (ix, iy, iz) to array indices (i, j, k).
        
        Parameters:
        - ix, iy, iz: World grid coordinate.
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
    def index_to_voxel_grid_coordinate(i: int, j: int, k: int , n: int) -> tuple[str]:
        """
        Convert array indices (i, j, k) back to world grid coordinate (ix, iy, iz).
        
        Parameters:
        - i, j, k: Array indices.
        - n: Size of the voxel world in each dimension.
        
        Returns:
        - ix, iy, iz: Corresponding world coordinate.
        """
        offset = (n - 1) // 2
        ix = i - offset
        iy = j - offset
        iz = k - 1 # No change needed for z as it cannot be negative.
        
        return ix, iy, iz