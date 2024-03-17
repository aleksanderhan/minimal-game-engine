import numpy as np
import numba as nb
from functools import lru_cache

from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

from constants import material_properties, voxel_type_map
from utils import index_to_voxel_grid_coordinates


def create_geometry(vertices: np.ndarray, indices: np.ndarray, name: str = "geom_node", debug: bool = False) -> NodePath:
    num_vertices = len(vertices) // 7
    vdata = GeomVertexData('voxel_data', _vertex_format_with_color(), Geom.UHStatic)
    vdata.setNumRows(num_vertices)

    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    normal_writer = GeomVertexWriter(vdata, 'normal')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Precompute colors if possible
    if debug:
        color_values = [(v[3], v[4], v[5], 0.8) for v in vertices.reshape(-1, 7)]
    else:
        color_values = [material_properties[voxel_type_map[int(v[6])]]["color"] for v in vertices.reshape(-1, 7)]

    for i, color in enumerate(color_values):
        idx = i * 7
        vertex_writer.addData3f(vertices[idx], vertices[idx+1], vertices[idx+2])
        normal_writer.addData3f(vertices[idx+3], vertices[idx+4], vertices[idx+5])
        color_writer.addData4f(*color)

    # Create triangles using indices
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(0, len(indices), 3):
        tris.addVertices(indices[i], indices[i+1], indices[i+2])
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)

    geom_node = GeomNode(name)
    geom_node.addGeom(geom)

    return NodePath(geom_node)

@nb.njit(nopython=True, cache=True)
def create_mesh(voxel_array: np.ndarray, exposed_voxels: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
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

@lru_cache
def _vertex_format_with_color() -> GeomVertexArrayFormat:
    # Define a vertex array format that includes position, normal, color, and texture
    array_format = GeomVertexArrayFormat()
    array_format.addColumn("vertex", 3, Geom.NTFloat32, Geom.CPoint)
    array_format.addColumn("normal", 3, Geom.NTFloat32, Geom.CVector)
    array_format.addColumn("color", 4, Geom.NTFloat32, Geom.CColor)

    # Create a vertex format based on the array format
    vertex_format = GeomVertexFormat()
    vertex_format.addArray(array_format)
    vertex_format = GeomVertexFormat.registerFormat(vertex_format)

    return vertex_format

@nb.njit(nopython=True, cache=True)
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

@nb.njit(nopython=True, cache=True)
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