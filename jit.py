import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def create_mesh(voxel_array: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Efficiently creates mesh data for exposed voxel faces.

    Args:
        voxel_world: 3D NumPy array representing voxel types.
        voxel_size: The size of each voxel in world units.

    Returns:
        vertices: A NumPy array of vertices where each group of six numbers represents the x, y, z coordinates of a vertex and its normal (nx, ny, nz) and color
        indices: A NumPy array of vertex indices, specifying how vertices are combined to form the triangular faces of the mesh.
    """
    vertices = []
    indices = []
    index_counter = 0  # Track indices for each exposed face

    exposed_voxels = identify_exposed_voxels(voxel_array)
    exposed_indices = np.argwhere(exposed_voxels)

    normals: dict[str, tuple[int, int, int]] = {
        (0, 1, 0): np.array([ 0.0,  1.0,  0.0]), # front
        (0, -1, 0): np.array([ 0.0, -1.0,  0.0]), # back
        (1, 0, 0): np.array([ 1.0,  0.0,  0.0]), # right
        (-1, 0, 0): np.array([-1.0,  0.0,  0.0]), # left
        (0, 0, 1): np.array([ 0.0,  0.0,  1.0]), # up
        (0, 0, -1): np.array([ 0.0,  0.0, -1.0]), # down
    }
    
    for i, j, k in exposed_indices:
        ix, iy, iz = index_to_voxel_grid_coordinates(i, j, k, voxel_array.shape[0])
        exposed_faces = _check_surrounding_air(voxel_array, i, j, k)

        for face_id, normal in normals.items():
            if face_id in exposed_faces:
                # Generate vertices for this face
                face_vertices = _generate_face_vertices(ix, iy, iz, face_id, voxel_size)

                # Generate normals for each face
                #face_normals = np.tile(np.array(normal), (4, 1))

                face_normals = [normal for _ in range(4)] # 4 normals per face
                voxel_type_value = voxel_array[i, j, k]

                # Append generated vertices, normals, and texture coordinates to the list
                for fv, fn in zip(face_vertices, face_normals):
                    vertices.extend([*fv, *fn, voxel_type_value])

                # Create indices for two triangles making up the face
                indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                    index_counter + 2, index_counter + 3, index_counter])
                
                index_counter += 4

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    return vertices, indices

@nb.jit(nopython=True, cache=True)
def _check_surrounding_air(array: np.ndarray, i: int, j: int, k: int) -> list[tuple[int, int]]:
    max_i, max_j, max_k = array.shape[0] - 1, array.shape[1] - 1, array.shape[2] - 1
    exposed_faces = []
    
    if j == max_j or array[i, j + 1, k] == 0: exposed_faces.append((0, 1, 0)) # front
    if j == 0 or array[i, j - 1, k] == 0: exposed_faces.append((0, -1, 0)) # back
    if i == max_i or array[i + 1, j, k] == 0: exposed_faces.append((1, 0, 0)) # right
    if i == 0 or array[i - 1, j, k] == 0: exposed_faces.append((-1, 0, 0)) # left
    if k == max_k or array[i, j, k + 1] == 0: exposed_faces.append((0, 0, 1)) # up
    if k == 0 or array[i, j, k - 1] == 0: exposed_faces.append((0, 0, -1)) # down
    
    return exposed_faces

@nb.jit(nopython=True, cache=True)
def _generate_face_vertices(ix: int, iy: int, iz: int, face_id: tuple[int, int, int], voxel_size: int) -> np.ndarray:
    """
    Generates vertices and normals for a given voxel face.

    Args:
        ix, iy, iz: coordinates of the voxel in the voxel grid.
        face_name: The face to be generated
        voxel_size: Size of the voxel.

    Returns:
        face_vertices: A list of vertex positions for the face.
    """
    face_offsets: dict[tuple[int, int, int], np.ndarray] = {
        (0, 1, 0): np.array([(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)]),
        (0, -1, 0): np.array([(0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5)]),
        (1, 0, 0): np.array([(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
        (-1, 0, 0): np.array([(-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5)]),
        (0, 0, 1): np.array([(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        (0, 0, -1): np.array([(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]),
    }[face_id]

    # Calculate the center position of the voxel in world coordinates
    center_position = np.array([ix, iy, iz]) * voxel_size

    # Then, for each face, adjust the vertices based on this center position
    return center_position + (face_offsets * voxel_size)

@nb.jit(nopython=True, cache=True)
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

@nb.jit(nopython=True, cache=True)
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

@nb.jit(nopython=True, cache=True)
def identify_exposed_voxels(voxel_array: np.ndarray) -> np.ndarray:
    """
    Identifies voxels exposed to air and returns a boolean array of the same shape as `voxel_array`
    indicating exposure. True means the voxel is exposed to air, False means it's not.

    Parameters:
        - voxel_array: a 3D numpy array representing voxel types as integers in the world.
    """
    # Create a new array with padding of 1 around the original array
    padded_shape = (voxel_array.shape[0] + 2, voxel_array.shape[1] + 2, voxel_array.shape[2] + 2)
    padded_world = np.zeros(padded_shape, dtype=voxel_array.dtype)
    
    # Fill the inner part of the padded array with the original voxel data
    padded_world[1:-1, 1:-1, 1:-1] = voxel_array
    
    # Initialize a boolean array for exposed faces
    exposed_faces = np.zeros_like(voxel_array, dtype=np.bool_)

    # Check all six directions
    exposed_faces |= ((padded_world[:-2, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[2:, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, :-2, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 2:, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, :-2] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, 2:] == 0) & (voxel_array > 0))
    
    return exposed_faces