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
    def check_surrounding_air(voxel_world, x, y, z):
        """
        Check each of the six directions around a point (x, y, z) in the voxel world
        for air (assumed to be represented by 0), including the boundary air of the world.
        """
        # Define the world's size
        max_x, max_y, max_z = voxel_world.shape[0] - 1, voxel_world.shape[1] - 1, voxel_world.shape[2] - 1

        # Initialize a list to store the names of faces exposed to air, including boundaries
        exposed_faces = []

        # Check each direction, directly considering the boundaries of the world
        if x == max_x or voxel_world[min(x + 1, max_x), y, z] == 0: exposed_faces.append("right")
        if x == 0 or voxel_world[max(x - 1, 0), y, z] == 0: exposed_faces.append("left")
        if y == max_y or voxel_world[x, min(y + 1, max_y), z] == 0: exposed_faces.append("front")
        if y == 0 or voxel_world[x, max(y - 1, 0), z] == 0: exposed_faces.append("back")
        if z == max_z or voxel_world[x, y, min(z + 1, max_z)] == 0: exposed_faces.append("up")
        if z == 0 or voxel_world[x, y, max(z - 1, 0)] == 0: exposed_faces.append("down")

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
    def identify_exposed_voxels(voxel_world):
        """
        Identifies a voxel exposed to air and returns a same shaped boolean np array with the result.
        True means it is exposed to air, False means it's not.

        Parameters:
            - voxel_world: a 3D numpy array representing the voxel types as integers in the world
        """
        # Pad the voxel world with zeros (air) on all sides
        padded_world = np.pad(voxel_world, pad_width=1, mode='constant', constant_values=0)
        
        exposed_faces = np.zeros_like(voxel_world, dtype=bool)
        
        for direction, (dx, dy, dz) in normals.items():
            shifted_world = np.roll(padded_world, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Expose face if there's air next to it (voxel value of 0 in the shifted world)
            exposed_faces |= ((shifted_world[1:-1, 1:-1, 1:-1] == 0) & (voxel_world > 0))
        
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
    def create_world_mesh(voxel_world, scale):
        """Efficiently creates mesh data for exposed voxel faces.

        Args:
            voxel_world: 3D NumPy array representing voxel types.
            scale: The size of each voxel in world units.

        Returns:
                vertices: A NumPy array of vertices where each group of six numbers represents the x, y, z coordinates of a vertex and its normal (nx, ny, nz).
                indices: A NumPy array of vertex indices, specifying how vertices are combined to form the triangular faces of the mesh.
        """

        exposed_voxels = VoxelTools.identify_exposed_voxels(voxel_world)

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        exposed_indices = np.argwhere(exposed_voxels)
        
        for i, j, k in exposed_indices:
            exposed_faces = VoxelTools.check_surrounding_air(voxel_world, i, j, k)
            c = 0
            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face
                    x, y, z = WorldTools.map_indices_to_coords(i, j, k, 0, 0, scale, voxel_world.shape[0])
                    face_vertices = VoxelTools.generate_face_vertices(x, y, z, face_name, scale)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type = voxel_world[i, j, k]
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
    def map_indices_to_coords(i, j, k, chunk_x, chunk_y, scale, chunk_size):
        print("i, j, k, chunk_x, chunk_y, scale, chunk_size", i, j, k, chunk_x, chunk_y, scale, chunk_size)
        """
        Converts local chunk indices (i, j, k) to global coordinates (x, y, z).

        Parameters:
            i, j, k: Local indices within a chunk.
            chunk_x, chunk_y: The chunk's position in the world.
            scale: The scale factor of the world, affecting the size of each voxel.
            chunk_size: The size of each chunk in terms of number of voxels.

        Returns:
            x, y, z: Global coordinates corresponding to the local indices.
        """
        # Convert local indices back to global coordinates
        # The formula takes the chunk's position in the world, multiplies by the chunk size and scale,
        # and then adds the local position within the chunk multiplied by the scale.
        x = (chunk_x * chunk_size + i) * scale * 2
        y = (chunk_y * chunk_size + j) * scale * 2
        z = k * scale * 2  # Assuming z starts at 0 and only positive values are considered

        return x, y, z

    
    @staticmethod
    def get_voxel_world(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y):
        if (chunk_x, chunk_y) not in voxel_world_map:
            width = chunk_size
            depth = chunk_size
            
            # Initialize an empty voxel world with air (0)
            voxel_world = np.zeros((width, depth, max_height), dtype=int)
            
            # Generate or retrieve heightmap for this chunk
            #heightmap = WorldTools.generate_flat_height_map(chunk_size, height=3)
            heightmap = WorldTools.generate_perlin_height_map(chunk_size, chunk_x, chunk_y)
            
            # Convert heightmap values to integer height levels, ensuring they do not exceed max_height
            height_levels = np.floor(heightmap).astype(int)
            height_levels = np.clip(height_levels, 1, max_height)
            adjusted_height_levels = height_levels[:-1, :-1]

            # Initialize the voxel world as zeros
            voxel_world = np.zeros((width, depth, max_height), dtype=int)

            # Create a 3D array representing each voxel's vertical index (Z-coordinate)
            z_indices = np.arange(max_height).reshape(1, 1, max_height)

            # Create a 3D boolean mask where true indicates a voxel should be set to rock (1)
            mask = z_indices < adjusted_height_levels[:,:,np.newaxis]

            # Apply the mask to the voxel world
            voxel_world[mask] = 1


            voxel_world = np.zeros((5, 5, 5), dtype=int)
            voxel_world[0, 0, 0] = 1    
            voxel_world[0, 0, 1] = 1
            voxel_world[-1, 0, 0] = 1



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
                height_map[x, y] = height * 30

        return height_map
    
    @staticmethod
    def generate_flat_height_map(board_size, height=1):
        # Adjust board_size to account for the extra row and column for seamless edges
        adjusted_size = board_size + 1
        # Create a 2D NumPy array filled with the specified height value
        height_map = np.full((adjusted_size, adjusted_size), height)
        return height_map