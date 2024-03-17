
# Toggle generator. Returns a or b alternatingly on next()
def toggle(a, b, yield_a=True):
    while True:
        (yield a) if yield_a else (yield b)
        yield_a = not yield_a


def voxel_grid_coordinates_to_index(ix: int, iy: int, iz: int, n: int) -> tuple[int, int, int]:
    """
    Convert world grid coordinates (ix, iy, iz) to array indices (i, j, k).
    
    Parameters:
    - ix, iy, iz: World grid coordinates.
    - n: Size of the voxel world in each dimension.
    
    Returns:
    - i, j, k: Corresponding array indices.
    """
    offset = (n - 1) // 2
    i = ix + offset
    j = iy + offset
    k = iz # No change needed for z as it cannot be negative.
    return i, j, k

def index_to_voxel_grid_coordinates(i: int, j: int, k: int , n: int) -> tuple[int, int, int]:
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
    iz = k # No change needed for z as it cannot be negative.
    
    return ix, iy, iz