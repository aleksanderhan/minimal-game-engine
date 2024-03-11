import numpy as np
import noise
import time

from panda3d.bullet import (
    BulletRigidBodyNode, BulletTriangleMesh, BulletTriangleMeshShape, BulletSphereShape
)
from panda3d.core import (
    Vec3
)

from constants import offset_arrays, normals, uv_maps
from world import VoxelWorld


# Toggle generator. Returns a or b alternatingly on next()
def toggle(a, b, yield_a=True):
    while True:
        (yield a) if yield_a else (yield b)
        yield_a = not yield_a


class DynamicArbitraryVoxelObject:

    def __init__(self, object_array, vertices, indices, mass=1, friction=1, name="object"):
        self.object_array = object_array
        self.vertices = vertices
        self.indices = indices
        self.name = name
        self.mass = mass
        self.friction = friction
        self.object_np = None
        self.object_node = None

    @staticmethod
    def make_single_voxel_object(scale: int, voxel_type: int, mass: int):
        object_array = np.zeros((1, 1, 1), dtype=int)
        vertices, indices = VoxelTools.create_voxel_mesh(voxel_type, scale)
        return DynamicArbitraryVoxelObject(object_array, vertices, indices, mass)
    

class VoxelTools:

    @staticmethod
    def create_dynamic_voxel_physics_node(object: DynamicArbitraryVoxelObject, scale: int):
        shape = BulletSphereShape(scale)
        node = BulletRigidBodyNode(object.name)
        node.setMass(object.mass)
        node.setFriction(object.friction)
        node.addShape(shape)
        return node
    
    @staticmethod
    def create_voxel_mesh(voxel_type, scale):
        vertices = []
        indices = []
        index_counter = 0

        j = 0
        for face_name, normal in normals.items():
            # Generate vertices for this face
            face_vertices = VoxelTools.generate_face_vertices(0, 0, 0, face_name, scale)
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
    def check_surrounding_air(voxel_world: VoxelWorld, i, j, k):
        max_i, max_j, max_k = voxel_world.shape()[0] - 1, voxel_world.shape()[1] - 1, voxel_world.shape()[2] - 1
        exposed_faces = []
        
        if i == max_i or voxel_world.world_array[i + 1, j, k] == 0: exposed_faces.append("right")
        if i == 0 or voxel_world.world_array[i - 1, j, k] == 0: exposed_faces.append("left")
        if j == max_j or voxel_world.world_array[i, j + 1, k] == 0: exposed_faces.append("front")
        if j == 0 or voxel_world.world_array[i, j - 1, k] == 0: exposed_faces.append("back")
        if k == max_k or voxel_world.world_array[i, j, k + 1] == 0: exposed_faces.append("up")
        if k == 0 or voxel_world.world_array[i, j, k - 1] == 0: exposed_faces.append("down")
        
        return exposed_faces
    
    @staticmethod
    def generate_face_vertices(x, y, z, face_name, scale):
        """
        Generates vertices and normals for a given voxel face.

        Args:
            x, y, z: Coordinates of the voxel in the voxel grid.
            face_name: The face to be generated
            scale: Size of the voxel.

        Returns:
            face_vertices: A list of vertex positions for the face.
        """
        face_offsets = offset_arrays[face_name]

        # Calculate vertex positions vectorized
        face_vertices = (np.array([x, y, z]) + face_offsets * scale).astype(float)

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
    def generate_chunk(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y, scale):
        t0 = time.perf_counter()
        voxel_world = WorldTools.get_voxel_world(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y)
        t1 = time.perf_counter()
        vertices, indices = WorldTools.create_world_mesh(voxel_world, scale)
        t2 = time.perf_counter()
        return vertices, indices, voxel_world, t1-t0, t2-t1
    
    @staticmethod
    def create_world_mesh(voxel_world: VoxelWorld, scale):
        """Efficiently creates mesh data for exposed voxel faces.

        Args:
            voxel_world: 3D NumPy array representing voxel types.
            scale: The size of each voxel in world units.

        Returns:
                vertices: A NumPy array of vertices where each group of six numbers represents the x, y, z coordinates of a vertex and its normal (nx, ny, nz).
                indices: A NumPy array of vertex indices, specifying how vertices are combined to form the triangular faces of the mesh.
        """

        exposed_voxels = VoxelTools.identify_exposed_voxels(voxel_world.get_world_array())

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        exposed_indices = np.argwhere(exposed_voxels)
        
        for i, j, k in exposed_indices:
            x, y, z = VoxelWorld.index_to_world(i, j, k, voxel_world.shape()[0])
            exposed_faces = VoxelTools.check_surrounding_air(voxel_world, i, j, k)

            c = 0
            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face

                    face_vertices = VoxelTools.generate_face_vertices(x, y, z, face_name, scale)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type = voxel_world.get_voxel(x, y, z)
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
    def get_voxel_world(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y):
        if (chunk_x, chunk_y) not in voxel_world_map:
            width = chunk_size
            depth = chunk_size
            
            # Initialize an empty voxel world with air (0)
            world_array = np.zeros((width, depth, max_height), dtype=int)
            
            # Generate or retrieve heightmap for this chunk
            #heightmap = WorldTools.generate_flat_height_map(chunk_size, height=3)
            heightmap = WorldTools.generate_perlin_height_map(chunk_size, chunk_x, chunk_y)
            
            # Convert heightmap values to integer height levels, ensuring they do not exceed max_height
            height_levels = np.floor(heightmap).astype(int)
            height_levels = np.clip(height_levels, 1, max_height)
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
            #world_array = np.zeros((5, 5, 5), dtype=int)
            voxel_world = VoxelWorld(world_array)

            #voxel_world.set_voxel(0, 0, 0, 1)
            #voxel_world.set_voxel(0, 0, 1, 1)
            #voxel_world.set_voxel(1, 0, 0, 1)
            #voxel_world.set_voxel(-1, 0, 0, 1)
            #voxel_world.set_voxel(0, 1, 0, 1)
            #voxel_world.set_voxel(0, -1, 0, 1)

            #voxel_world.set_voxel(1, -1, 0, 1)
            #voxel_world.set_voxel(-1, -1, 0, 1)
            #voxel_world.set_voxel(1, 1, 0, 1)
            #voxel_world.set_voxel(-1, 1, 0, 1)



            voxel_world_map[(chunk_x, chunk_y)] = voxel_world
            return voxel_world

        return voxel_world_map.get((chunk_x, chunk_y)) 

    @staticmethod
    def generate_perlin_height_map(chunk_size, chunk_x, chunk_y):
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
    def generate_flat_height_map(board_size, height=1):
        # Adjust board_size to account for the extra row and column for seamless edges
        adjusted_size = board_size + 1
        # Create a 2D NumPy array filled with the specified height value
        height_map = np.full((adjusted_size, adjusted_size), height)
        return height_map