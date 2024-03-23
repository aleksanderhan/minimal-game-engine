import numpy as np
from functools import lru_cache

from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

from jit import _create_mesh
from util import create_voxel_type_value_color_list


def create_mesh(voxel_array: np.ndarray, 
                voxel_size: float,
                debug: bool) -> tuple[np.ndarray, np.ndarray]:
    
    voxel_type_value_color_list = create_voxel_type_value_color_list()
    return _create_mesh(voxel_array, voxel_size, voxel_type_value_color_list, debug)    

def create_geometry(vertices: np.ndarray, indices: np.ndarray, name: str = "geom_node") -> NodePath:
    num_vertices = len(vertices) // 10
    vdata = GeomVertexData('voxel_data', _vertex_format_with_color(), Geom.UHStatic)
    vdata.uncleanSetNumRows(num_vertices)

    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    normal_writer = GeomVertexWriter(vdata, 'normal')
    color_writer = GeomVertexWriter(vdata, 'color')

    for i in range(num_vertices):
        base_idx = i * 10  # 3 for pos, 3 for normal, 4 for color = 10 components per vertex
        vertex_writer.addData3f(vertices[base_idx], vertices[base_idx + 1], vertices[base_idx + 2])
        normal_writer.addData3f(vertices[base_idx + 3], vertices[base_idx + 4], vertices[base_idx + 5])
        color_writer.addData4f(vertices[base_idx + 6], vertices[base_idx + 7], vertices[base_idx + 8], vertices[base_idx + 9])

    # Create triangles using indices
    tris = GeomTriangles(Geom.UHStatic)
    tris.reserve_num_vertices(len(indices))
    for i in range(0, len(indices), 3):
        tris.addVertices(indices[i], indices[i+1], indices[i+2])
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)

    geom_node = GeomNode(name)
    geom_node.addGeom(geom)

    return NodePath(geom_node)

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
