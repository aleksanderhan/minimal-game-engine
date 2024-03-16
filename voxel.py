import numpy as np
import uuid
from functools import lru_cache

from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.core import Vec3, LQuaternionf
from panda3d.bullet import BulletGenericConstraint, BulletRigidBodyNode
from panda3d.core import TransformState, NodePath

from constants import offset_arrays, normals, material_properties, VoxelType, voxel_type_map
from misc_utils import IndexTools


class DynamicArbitraryVoxelObject:

    def __init__(self, 
                 voxel_array: np.ndarray, 
                 voxel_size: float, 
                 vertices: np.ndarray, 
                 indices: np.ndarray, 
                 mass: float, 
                 friction: float, 
                 name: str = "VoxelObject"):
        
        self.root = self
        self.parent = self.root
        self.parent_position_relative_to_grid = Vec3(0, 0, 0)

        self.voxel_array = voxel_array
        self.voxel_size = voxel_size
        self.vertices = vertices
        self.indices = indices
        
        self.name = name
        self.id = str(uuid.uuid4())
        self.mass = mass
        self.friction = friction
        
        self.node_paths = {}

    def get_offset(self) -> np.ndarray: # deprecated?
        return (np.array(self.voxel_array.shape) - 1) // 2

    def add_voxel(self, hit_pos: Vec3, hit_normal: Vec3, voxel_type: VoxelType, game_engine):
        root_node_path = self.node_paths[(0, 0, 0)]
        root_node_pos = root_node_path.getPos()       
        orientation = root_node_path.getQuat()

        # Use the quaternion to transform the local forward direction to world space
        world_forward = Vec3(0, 1, 0)
        local_forward = orientation.xform(world_forward)

        ix = 1# TODO
        iy = 1# TODO
        iz = 1# TODO


        max_i, max_j, max_k = self.voxel_array.shape
        if not (0 <= ix < max_i and 0 <= iy < max_j and 0 <= iz < max_k):
            self.extend_array_uniformly()
        self.voxel_array[ix, iy, iz] = voxel_type.value

        self.vertices, self.indices = VoxelTools.create_object_mesh(self)

        body_indices = np.argwhere(self.voxel_array)
        parts = []
        for i, j, k in body_indices:
            ix, iy, iz = IndexTools.index_to_voxel_grid_coordinates(i, j, k, self.voxel_array.shape[0])

            position = root_node_pos

            node_np = VoxelTools.create_dynamic_single_voxel_physics_node(self, game_engine.render, game_engine.physics_world)
            node_np.setPythonTag("object", self)
            node_np.setQuat(orientation)
            node_np.setPos(position)
            self.node_paths[(i, j, k)] = node_np
            parts.append(node_np)

        for i in range(1, len(parts)):
            # Assuming bodyA and bodyB are instances of BulletRigidBodyNode you want to connect
            constraint = BulletGenericConstraint(parts[i-1].node(), parts[i].node(), TransformState.makeIdentity(), TransformState.makeIdentity(), True)

            # Lock all degrees of freedom to simulate a fixed constraint
            for i in range(6):
                constraint.setParam(i, 0, 0) 
            game_engine.physics_world.attachConstraint(constraint)

        return root_node_pos, orientation

    def extend_array_uniformly(self):
        # Specify the padding width of 1 for all sides of all dimensions
        pad_width = [(1, 1)] * 3  # Padding for depth, rows, and columns
        
        # Pad the array with 0's on all sides
        self.voxel_array = np.pad(self.voxel_array, pad_width=pad_width, mode='constant', constant_values=0)

    def get_pos(self, render):
        # TODO: somehow calculate position - center of all the nodes? origin voxel?
        # To get the position relative to the parent node
        origin_node = self.node_paths[(0, 0)]
        local_position = origin_node.getPos()

        # To get the global position (relative to the world or render node)
        global_position = origin_node.getPos(render)

        print(f"Local Position: {local_position}")
        print(f"Global Position: {global_position}")

    def get_orientation(self) -> LQuaternionf:
        root_node_path = self.node_paths[(0, 0, 0)]
        return root_node_path.getQuat()
    
    
class VoxelTools:        

    @staticmethod
    def crate_dynamic_single_voxel_object(voxel_size: int, voxel_type: VoxelType, render, physics_world) -> DynamicArbitraryVoxelObject:
        voxel_array = np.ones((1, 1, 1), dtype=int)
        vertices, indices = VoxelTools.create_single_voxel_mesh(voxel_type, voxel_size)

        mass = material_properties[voxel_type]["mass"]
        friction = material_properties[voxel_type]["friction"]
        
        object = DynamicArbitraryVoxelObject(voxel_array, voxel_size, vertices, indices, mass, friction)
        object_np = VoxelTools.create_dynamic_single_voxel_physics_node(object, render, physics_world)
        object_np.setPythonTag("object", object)
        object_np.setPythonTag("ijk", (0, 0, 0))
        object.node_paths[(0, 0, 0)]  = object_np
        return object            

    @staticmethod
    def create_dynamic_single_voxel_physics_node(object: DynamicArbitraryVoxelObject, render, physics_world) -> NodePath:
        print(type(render), type(physics_world)) # to add types hints
        voxel_type_value = object.voxel_array[0, 0, 0]
        voxel_type = voxel_type_map[voxel_type_value]
        material = material_properties[voxel_type]
        
        radius = object.voxel_size / 2
        shape = BulletSphereShape(radius)
        node = BulletRigidBodyNode(object.name)
        node.setMass(material["mass"])
        node.setFriction(material["friction"])
        node.addShape(shape)

        # Set the initial position of the segment
        node_np = render.attachNewNode(node)
        
        # Add the node to the Bullet world
        physics_world.attachRigidBody(node)
        return node_np
    
    @lru_cache
    @staticmethod
    def create_single_voxel_mesh(voxel_type: VoxelType, voxel_size: int) -> tuple[np.ndarray, np.ndarray]:
        vertices = []
        indices = []
        index_counter = 0

        j = 0
        for face_name, normal in normals.items():
            # Generate vertices for this face
            face_vertices = VoxelTools.generate_face_vertices(0, 0, 0, face_name, voxel_size)
            face_normals = np.tile(np.array(normal), (4, 1))

            # Append generated vertices, normals, and texture coordinates to the list
            for fv, fn in zip(face_vertices, face_normals):
                vertices.extend([*fv, *fn, voxel_type.value])
            
            # Create indices for two triangles making up the face
            indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                index_counter + 2, index_counter + 3, index_counter])
            
            index_counter += 4
            j += 1

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)
    
    @staticmethod
    def create_object_mesh(object: DynamicArbitraryVoxelObject) -> tuple[np.ndarray, np.ndarray]:
        exposed_voxels = VoxelTools.identify_exposed_voxels(object.voxel_array)

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        exposed_indices = np.argwhere(exposed_voxels)
        
        for i, j, k in exposed_indices:
            ix, iy, iz = IndexTools.index_to_voxel_grid_coordinates(i, j, k, object.voxel_array.shape[0])
            exposed_faces = VoxelTools.check_surrounding_air(object.voxel_array, i, j, k)

            c = 0
            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face

                    face_vertices = VoxelTools.generate_face_vertices(ix, iy, iz, face_name, object.voxel_size)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type_value = object.voxel_array[i, j, k]

                    # Append generated vertices, normals, and color to the list
                    for fv, fn in zip(face_vertices, face_normals):
                        vertices.extend([*fv, *fn, voxel_type_value])
                    
                    # Create indices for two triangles making up the face
                    indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                        index_counter + 2, index_counter + 3, index_counter])
                    
                    index_counter += 4
                    c += 1

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    @staticmethod
    def check_surrounding_air(array: np.ndarray, i: int, j: int, k: int) -> list[str]:
        max_i, max_j, max_k = array.shape[0] - 1, array.shape[1] - 1, array.shape[2] - 1
        exposed_faces = []
        
        if j == max_j or array[i, j + 1, k] == 0: exposed_faces.append("front")
        if j == 0 or array[i, j - 1, k] == 0: exposed_faces.append("back")
        if i == max_i or array[i + 1, j, k] == 0: exposed_faces.append("right")
        if i == 0 or array[i - 1, j, k] == 0: exposed_faces.append("left")
        if k == max_k or array[i, j, k + 1] == 0: exposed_faces.append("up")
        if k == 0 or array[i, j, k - 1] == 0: exposed_faces.append("down")
        
        return exposed_faces
    
    @lru_cache
    @staticmethod
    def generate_face_vertices(ix: int, iy: int, iz: int, face_name: str, voxel_size: int):
        """
        Generates vertices and normals for a given voxel face.

        Args:
            ix, iy, iz: coordinates of the voxel in the voxel grid.
            face_name: The face to be generated
            voxel_size: Size of the voxel.

        Returns:
            face_vertices: A list of vertex positions for the face.
        """
        face_offsets = offset_arrays[face_name]   

        # Calculate the center position of the voxel in world coordinates
        center_position = np.array([ix, iy, iz]) * voxel_size

        # Then, for each face, adjust the vertices based on this center position
        face_vertices = center_position + (face_offsets * voxel_size)

        return face_vertices
    
    @staticmethod
    def noop_transform(face):
        return face
    
    @staticmethod
    def rotate_face_90_degrees_ccw_around_z(face):
        # Rotate each point in the face 90 degrees counter-clockwise around the Z axis
        return [(y, -x, z) for x, z, y in face]
    
    @staticmethod
    def rotate_face_90_degrees_ccw_around_x(face):
        # Rotate each point in the face 90 degrees counter-clockwise around the X axis
        return [(x, -z, y) for x, y, z in face]

    @staticmethod
    def rotate_face_90_degrees_ccw_around_y(face):
        # Rotate each point in the face 90 degrees counter-clockwise around the Y axis
        return [(z, y, -x) for x, y, z in face]

    @staticmethod
    def identify_exposed_voxels(world_array: np.ndarray) -> np.ndarray:
        """
        Identifies a voxel exposed to air and returns a same shaped boolean np array with the result.
        True means it is exposed to air, False means it's not.

        Parameters:
            - world_array: a 3D numpy array representing the voxel types as integers in the world
        """
        # Pad the voxel world with zeros (air) on all sides
        padded_world = np.pad(world_array, pad_width=1, mode='constant', constant_values=0)
        
        exposed_faces = np.zeros_like(world_array, dtype=bool)
        
        for direction, (dx, dy, dz) in normals.items():
            shifted_world = np.roll(padded_world, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Expose face if there's air next to it (voxel value of 0 in the shifted world)
            exposed_faces |= ((shifted_world[1:-1, 1:-1, 1:-1] == 0) & (world_array > 0))
        
        return exposed_faces
