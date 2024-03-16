import numpy as np
from functools import lru_cache

from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

from constants import material_properties, voxel_type_map


class GeometryTools:

    #@lru_cache # - unhashable type: 'numpy.ndarray'
    @staticmethod
    def create_geometry(vertices: np.ndarray, indices: np.ndarray, name: str = "geom_node", debug: bool = False) -> NodePath:
        vdata = GeomVertexData('voxel_data', GeometryTools.vertex_format_with_color(), Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        normal_writer = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')

        for i in range(0, len(vertices), 7):  # 7 components per vertex: 3 position, 3 normal, 1 voxel_type
            vertex_writer.addData3f(vertices[i], vertices[i+1], vertices[i+2])
            normal_writer.addData3f(vertices[i+3], vertices[i+4], vertices[i+5])

            if debug:
                dx, dy, dz = vertices[i+3], vertices[i+4], vertices[i+5]
                color = (dx, dy, dz, 0.8) #color_normal_map[(dx, dy, dz)]
            else:
                voxel_type_value = int(vertices[i+6])
                voxel_type = voxel_type_map[voxel_type_value]
                color = material_properties[voxel_type]["color"]
            #print(color)
            color_writer.addData4f(color)

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
    @staticmethod
    def vertex_format_with_color() -> GeomVertexArrayFormat:
        # Define a vertex array format that includes position, normal, color, and texture
        array_format = GeomVertexArrayFormat()
        array_format.addColumn("vertex", 3, Geom.NTFloat32, Geom.CPoint)
        array_format.addColumn("normal", 3, Geom.NTFloat32, Geom.CVector)
        array_format.addColumn("color", 4, Geom.NTFloat32, Geom.CColor)
        array_format.addColumn("texcoord", 2, Geom.NTFloat32, Geom.CTexcoord)

        # Create a vertex format based on the array format
        vertex_format = GeomVertexFormat()
        vertex_format.addArray(array_format)
        vertex_format = GeomVertexFormat.registerFormat(vertex_format)

        return vertex_format