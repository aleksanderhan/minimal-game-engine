import numpy as np
import numba as nb
from functools import lru_cache

from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

from constants import material_properties, voxel_type_map


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
