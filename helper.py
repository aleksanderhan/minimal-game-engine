import numpy as np
import noise
import time
import uuid
import math

from panda3d.bullet import (
    BulletRigidBodyNode, BulletTriangleMesh, BulletTriangleMeshShape, BulletSphereShape
)
from panda3d.core import (
    Vec3, LQuaternionf
)
from panda3d.bullet import BulletGenericConstraint, BulletRigidBodyNode
from panda3d.core import TransformState


from constants import offset_arrays, normals, uv_maps, material_properties, VoxelType, voxel_type_map
from world import VoxelWorld


# Toggle generator. Returns a or b alternatingly on next()
def toggle(a, b, yield_a=True):
    while True:
        (yield a) if yield_a else (yield b)
        yield_a = not yield_a


class DynamicArbitraryVoxelObject:

    def __init__(self, voxel_array, voxel_size, vertices, indices, mass, friction, name="VoxelObject"):
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

    def get_offset(self):
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
            ix, iy, iz = VoxelWorld.index_to_voxel_grid_coordinates(i, j, k, self.voxel_array.shape[0])

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



        print()
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
    def create_dynamic_single_voxel_physics_node(object: DynamicArbitraryVoxelObject, render, physics_world):
        voxel_type_int = object.voxel_array[0, 0, 0]
        voxel_type = voxel_type_map[voxel_type_int]
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
    
    @staticmethod
    def create_single_voxel_mesh(voxel_type: VoxelType, voxel_size):
        vertices = []
        indices = []
        index_counter = 0

        j = 0
        for face_name, normal in normals.items():
            # Generate vertices for this face
            face_vertices = VoxelTools.generate_face_vertices(0, 0, 0, face_name, voxel_size)
            face_normals = np.tile(np.array(normal), (4, 1))

            uvs = uv_maps[voxel_type][face_name]
            u, v = uvs[j % 4]  # Cycle through the UV coordinates for each vertex

            # Append generated vertices, normals, and texture coordinates to the list
            for fv, fn in zip(face_vertices, face_normals):
                vertices.extend([*fv, *fn, u, v])
            
            # Create indices for two triangles making up the face
            indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                index_counter + 2, index_counter + 3, index_counter])
            
            index_counter += 4
            j += 1

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)
    
    @staticmethod
    def create_object_mesh(object: DynamicArbitraryVoxelObject):
        exposed_voxels = VoxelTools.identify_exposed_voxels(object.voxel_array)

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        exposed_indices = np.argwhere(exposed_voxels)
        
        for i, j, k in exposed_indices:
            ix, iy, iz = VoxelWorld.index_to_voxel_grid_coordinates(i, j, k, object.voxel_array.shape[0])
            exposed_faces = VoxelTools.check_surrounding_air(object.voxel_array, i, j, k)

            c = 0
            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face

                    face_vertices = VoxelTools.generate_face_vertices(ix, iy, iz, face_name, object.voxel_size)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type_int = object.voxel_array[i, j, k]
                    voxel_type = voxel_type_map[voxel_type_int]
                    uvs = uv_maps[voxel_type][face_name]
                    u, v = uvs[c % 4]  # Cycle through the UV coordinates for each vertex

                    # Append generated vertices, normals, and texture coordinates to the list
                    for fv, fn in zip(face_vertices, face_normals):
                        vertices.extend([*fv, *fn, u, v])
                    
                    # Create indices for two triangles making up the face
                    indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                        index_counter + 2, index_counter + 3, index_counter])
                    
                    index_counter += 4
                    c += 1

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    @staticmethod
    def check_surrounding_air(array, i, j, k):
        max_i, max_j, max_k = array.shape[0] - 1, array.shape[1] - 1, array.shape[2] - 1
        exposed_faces = []
        
        if j == max_j or array[i, j + 1, k] == 0: exposed_faces.append("front")
        if j == 0 or array[i, j - 1, k] == 0: exposed_faces.append("back")
        if i == max_i or array[i + 1, j, k] == 0: exposed_faces.append("right")
        if i == 0 or array[i - 1, j, k] == 0: exposed_faces.append("left")
        if k == max_k or array[i, j, k + 1] == 0: exposed_faces.append("up")
        if k == 0 or array[i, j, k - 1] == 0: exposed_faces.append("down")
        
        return exposed_faces
    
    @staticmethod
    def generate_face_vertices(ix, iy, iz, face_name, voxel_size):
        """
        Generates vertices and normals for a given voxel face.

        Args:
            ix, iy, iz: Coordinates of the voxel in the voxel grid.
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
    def identify_exposed_voxels(world_array):
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
    

class WorldTools:

    @staticmethod   
    def get_center_of_hit_static_voxel(raycast_result, voxel_size: float) -> Vec3:
        """
        Identifies the voxel hit by a raycast and returns its center position in world space.
        
        Parameters:
        - raycast_result: The raycast object with a hit.
        - voxel_size: The size of a voxel.
        
        Returns:
        - Vec3: The center position of the hit voxel in world space.
        """
        hit_pos = raycast_result.getHitPos()
        hit_normal = raycast_result.getHitNormal()

        if hit_normal == Vec3(0, -1, 0) or hit_normal == Vec3(0, 1, 0) or hit_normal == Vec3(1, 0, 0) or hit_normal == Vec3(-1, 0, 0): # all side faces
            hit_pos.z += voxel_size / 2

        if  hit_normal == Vec3(0, 1, 0): # Front face
            hit_pos.y -= voxel_size/ 2

        if hit_normal == Vec3(1, 0, 0): # right face
            hit_pos.x -= voxel_size/ 2

        if hit_normal == Vec3(-1, 0, 0): # left face
            hit_pos.x += voxel_size/ 2

        if hit_normal == Vec3(0, 0, -1): # bottom face
            hit_pos.z += voxel_size

        if hit_normal == Vec3(0, -1, 0): # back face
            hit_pos.y += voxel_size / 2


        # Define a minimal bias to adjust hit positions near voxel boundaries
        bias = voxel_size * 0.01  # 1% of the voxel size

        # Apply bias based on the sign of the hit position components
        biased_x = hit_pos.x + bias if hit_pos.x > 0 else hit_pos.x - bias
        biased_y = hit_pos.y + bias if hit_pos.y > 0 else hit_pos.y - bias
        biased_z = hit_pos.z + bias if hit_pos.z > 0 else hit_pos.z - bias

        # Convert the biased hit position to voxel grid coordinates
        grid_x = round(biased_x / voxel_size)
        grid_y = round(biased_y / voxel_size)
        grid_z = round(biased_z / voxel_size)

        # Calculate the center of the voxel in world coordinates
        voxel_center_x = grid_x * voxel_size
        voxel_center_y = grid_y * voxel_size
        voxel_center_z = grid_z * voxel_size - voxel_size / 2  # Adjust for top face at z=0

        return Vec3(voxel_center_x, voxel_center_y, voxel_center_z)
    
    @staticmethod
    def get_center_of_hit_dynamic_voxel(raycast_result) -> Vec3:
        hit_node = raycast_result.getNode()
        hit_object = hit_node.getPythonTag("object")
        ijk = hit_node.getPythonTag("ijk")

        node_np = hit_object.node_paths[ijk]
        return node_np.getPos()
    
    @staticmethod
    def generate_chunk(chunk_size: int, max_height: int, voxel_world_map: dict, chunk_x: int, chunk_y: int, voxel_size: float):
        t0 = time.perf_counter()
        voxel_world = WorldTools.get_voxel_world(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y, voxel_size)
        t1 = time.perf_counter()
        vertices, indices = WorldTools.create_world_mesh(voxel_world, voxel_size)
        t2 = time.perf_counter()
        return vertices, indices, voxel_world, t1-t0, t2-t1
    
    @staticmethod
    def create_world_mesh(voxel_world: VoxelWorld, voxel_size: float):
        """Efficiently creates mesh data for exposed voxel faces.

        Args:
            voxel_world: 3D NumPy array representing voxel types.
            voxel_size: The size of each voxel in world units.

        Returns:
            vertices: A NumPy array of vertices where each group of six numbers represents the x, y, z coordinates of a vertex and its normal (nx, ny, nz).
            indices: A NumPy array of vertex indices, specifying how vertices are combined to form the triangular faces of the mesh.
        """

        exposed_voxels = VoxelTools.identify_exposed_voxels(voxel_world.world_array)

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        exposed_indices = np.argwhere(exposed_voxels)
        
        for i, j, k in exposed_indices:
            ix, iy, iz = VoxelWorld.index_to_voxel_grid_coordinates(i, j, k, voxel_world.world_array.shape[0])
            exposed_faces = VoxelTools.check_surrounding_air(voxel_world.world_array, i, j, k)

            c = 0
            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face

                    face_vertices = VoxelTools.generate_face_vertices(ix, iy, iz, face_name, voxel_size)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type = voxel_world.get_voxel_type(ix, iy, iz)
                    uvs = uv_maps[voxel_type][face_name]
                    u, v = uvs[c % 4]  # Cycle through the UV coordinates for each vertex

                    # Append generated vertices, normals, and texture coordinates to the list
                    for fv, fn in zip(face_vertices, face_normals):
                        vertices.extend([*fv, *fn, u, v])
                    
                    # Create indices for two triangles making up the face
                    indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                        index_counter + 2, index_counter + 3, index_counter])
                    
                    index_counter += 4
                    c += 1

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)
    
    @staticmethod
    def get_voxel_world(chunk_size: int, max_height: int, voxel_world_map: dict, chunk_x: int, chunk_y: int, voxel_size: float):
        if (chunk_x, chunk_y) not in voxel_world_map:
            width = chunk_size
            depth = chunk_size
            
            # Initialize an empty voxel world with air (0)
            world_array = np.zeros((width, depth, max_height), dtype=int)
            
            # Generate or retrieve heightmap for this chunk
            heightmap = WorldTools.generate_flat_height_map(chunk_size, height=1)
            #heightmap = WorldTools.generate_perlin_height_map(chunk_size, chunk_x, chunk_y)
            
            # Convert heightmap values to integer height levels, ensuring they do not exceed max_height
            height_levels = np.floor(heightmap).astype(int)
            height_levels = np.clip(height_levels, 0, max_height)
            adjusted_height_levels = height_levels[:-1, :-1]

            # Create a 3D array representing each voxel's vertical index (Z-coordinate)
            z_indices = np.arange(max_height).reshape(1, 1, max_height)

            # Create a 3D boolean mask where true indicates a voxel should be set to rock (1)
            mask = z_indices < adjusted_height_levels[:,:,np.newaxis]

            # Apply the mask to the voxel world
            world_array[mask] = 1



            '''
            world_array = np.array([[[0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]])
            '''
            world_array = np.zeros((5, 5, 5), dtype=int)
            voxel_world = VoxelWorld(world_array, voxel_size)

            #if chunk_x == 0 and chunk_y == 0:
            #    voxel_world.set_voxel(0, 0, 1, VoxelType.STONE)
            
            
            voxel_world.set_voxel(0, 0, 0, VoxelType.STONE)
            #voxel_world.set_voxel(1, 0, 0, VoxelType.STONE)
            #voxel_world.set_voxel(-1, 0, 0, VoxelType.STONE)
            #voxel_world.set_voxel(0, 1, 0, VoxelType.STONE)
            #voxel_world.set_voxel(0, -1, 0, VoxelType.STONE)
            #voxel_world.set_voxel(1, -1, 0, VoxelType.STONE)
            #voxel_world.set_voxel(-1, -1, 0, VoxelType.STONE)
            #voxel_world.set_voxel(1, 1, 0,VoxelType.STONE)
            #voxel_world.set_voxel(-1, 1, 0, VoxelType.STONE)



            voxel_world_map[(chunk_x, chunk_y)] = voxel_world
            return voxel_world

        return voxel_world_map.get((chunk_x, chunk_y)) 

    @staticmethod
    def generate_perlin_height_map(chunk_size: int, chunk_x: int, chunk_y: int):
        scale = 0.05  # Adjust scale to control the "zoom" level of the noise
        octaves = 6  # Number of layers of noise to combine
        persistence = 0.5  # Amplitude of each octave
        lacunarity = 2.0  # Frequency of each octave

        height_map = np.zeros((chunk_size + 1, chunk_size + 1))

        # Calculate global offsets
        global_offset_x = chunk_x * chunk_size
        global_offset_y = chunk_y * chunk_size

        for x in range(chunk_size + 1):
            for y in range(chunk_size + 1):
                # Calculate global coordinates
                global_x = (global_offset_x + x) * scale
                global_y = (global_offset_y + y) * scale

                # Generate height using Perlin noise
                height = noise.pnoise2(global_x, global_y,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=10000,  # Large repeat region to avoid repetition
                                    repeaty=10000,
                                    base=1)  # Base can be any constant, adjust for different terrains

                # Map the noise value to a desired height range if needed
                height_map[x, y] = height * 20

        return height_map
    
    @staticmethod
    def generate_flat_height_map(board_size: int, height: int=1):
        # Adjust board_size to account for the extra row and column for seamless edges
        adjusted_size = board_size + 1
        # Create a 2D NumPy array filled with the specified height value
        height_map = np.full((adjusted_size, adjusted_size), height)
        return height_map
    
    @staticmethod
    def calculate_chunk_world_position(chunk_x: int, chunk_y: int, chunk_size: int, voxel_size: float):
        """
        Calculates the world position of the chunk origo based on its grid coordinates.

        Parameters:
        - chunk_x, chunk_y: The chunk's position in the grid/map.
        - chunk_size: Number of voxels along a single axis of the chunk.
        - voxel_size: Size of a voxel

        Returns:
        Tuple[float, float]: The world coordinates of the chunk.
        """
        x = chunk_x * chunk_size * voxel_size
        y = chunk_y * chunk_size * voxel_size

        return x, y
    
    @staticmethod
    def calculate_world_chunk_position(position: Vec3, chunk_size: int, voxel_size: float):
        """
        Calculates the chunk grid coordinates corresponding to a world position.

        Parameters:
        - position: A Vec3 representing the world position.
        - voxel_size: Size of a voxel.
        - chunk_size: Number of voxels along a single axis of the chunk.

        Returns:    
        Tuple[int, int]: The chunk's grid coordinates (chunk_x, chunk_y).
        """
        # Calculate the half-size of a chunk in world units
        half_chunk_size_world_units = (chunk_size * voxel_size) / 2

        # Adjust position to align chunk centers with the origin
        adjusted_pos_x = position.x + half_chunk_size_world_units
        adjusted_pos_y = position.y + half_chunk_size_world_units

        # Calculate chunk coordinates
        chunk_x = math.floor(adjusted_pos_x / (chunk_size * voxel_size)) if adjusted_pos_x >= 0 else math.ceil(adjusted_pos_x / (chunk_size * voxel_size)) - 1
        chunk_y = math.floor(adjusted_pos_y / (chunk_size * voxel_size)) if adjusted_pos_y >= 0 else math.ceil(adjusted_pos_y / (chunk_size * voxel_size)) - 1

        return chunk_x, chunk_y