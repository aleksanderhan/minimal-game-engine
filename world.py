import numpy as np
import noise
import math
from functools import lru_cache

from panda3d.core import Vec3, Vec2, NodePath
from panda3d.bullet import BulletClosestHitRayResult, BulletRigidBodyNode

from constants import VoxelType, voxel_type_map, normals
from voxel import VoxelTools
from misc_utils import IndexTools


class VoxelWorld:

    def __init__(self, world_array: np.ndarray, voxel_size: int):

        self.voxel_size = voxel_size
        self.world_array = world_array
        self.chunk_coord: tuple[int, int] = None
        
        self.vertices: np.ndarray = None
        self.indices: np.ndarray = None
        self.terrain_np: NodePath = None
        self.terrain_node: BulletRigidBodyNode = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VoxelWorld):
            return np.array_equal(self.world_array, other.world_array) and self.chunk_coord == other.chunk_coord \
                and self.voxel_size == other.voxel_size and self.chunk_coord == other.chunk_coord
        
        return False
    
    def __hash__(self):
        return hash((self.chunk_coord, self.world_array, self.voxel_size))
        
    def get_voxel_type(self, ix: int, iy: int, iz: int) -> VoxelType: 
        # Assuming world_array is centered around (0, 0, 0) at initialization
        # and offset is half the size of the current array dimensions.
        i, j, k = IndexTools.voxel_grid_coordinate_to_index(ix, iy, iz, self.world_array.shape[0])
        voxel_type_int = self.world_array[i, j, k]
        return voxel_type_map[voxel_type_int]

    def set_voxel(self, ix: int, iy: int, iz: int, voxel_type: VoxelType):
        i, j, k = IndexTools.voxel_grid_coordinate_to_index(ix, iy, iz, self.world_array.shape[0])
        self.world_array[i, j, k] = voxel_type.value

    def create_world_mesh(self):
        """Efficiently creates mesh data for exposed voxel faces.

        Args:
            voxel_world: 3D NumPy array representing voxel types.
            voxel_size: The size of each voxel in world units.

        Returns:
            vertices: A NumPy array of vertices where each group of six numbers represents the x, y, z coordinate of a vertex and its normal (nx, ny, nz).
            indices: A NumPy array of vertex indices, specifying how vertices are combined to form the triangular faces of the mesh.
        """

        exposed_voxels = VoxelTools.identify_exposed_voxels(self.world_array)

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        exposed_indices = np.argwhere(exposed_voxels)
        
        for i, j, k in exposed_indices:
            ix, iy, iz = IndexTools.index_to_voxel_grid_coordinate(i, j, k, self.world_array.shape[0])
            exposed_faces = VoxelTools.check_surrounding_air(self.world_array, i, j, k)

            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face
                    face_vertices = VoxelTools.generate_face_vertices(ix, iy, iz, face_name, self.voxel_size)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type = self.get_voxel_type(ix, iy, iz)

                    # Append generated vertices, normals, and texture coordinate to the list
                    for fv, fn in zip(face_vertices, face_normals):
                        vertices.extend([*fv, *fn, voxel_type.value])
                    
                    # Create indices for two triangles making up the face
                    indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                        index_counter + 2, index_counter + 3, index_counter])
                    
                    index_counter += 4

        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)

   

class WorldTools:

    @staticmethod
    @lru_cache
    def calculate_distance_between_2d_points(point1: tuple[int, int] | Vec2, point2: tuple[int, int] | Vec2) -> float:
        point1_x, point1_y = point1
        point2_x, point2_y = point2
        return ((point2_x - point1_x)**2 + (point2_y - point1_y)**2)**0.5

    @staticmethod   
    def get_center_of_hit_static_voxel(raycast_result: BulletClosestHitRayResult, voxel_size: float) -> Vec3:
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

        # Nudge the hit position slightly towards the opposite direction of the normal
        # This helps ensure the hit position is always considered within the voxel, even near edges
        nudge = hit_normal * (voxel_size * 0.015)  # Nudge by 1.5% of the voxel size towards the voxel center
        adjusted_hit_pos = hit_pos - nudge

        # Convert the adjusted hit position to voxel grid coordinate
        grid_x = round(adjusted_hit_pos.x / voxel_size)
        grid_y = round(adjusted_hit_pos.y / voxel_size)
        grid_z = round(adjusted_hit_pos.z / voxel_size)

        # Calculate the center of the voxel in world coordinate
        voxel_center_x = grid_x * voxel_size
        voxel_center_y = grid_y * voxel_size
        voxel_center_z = grid_z * voxel_size - voxel_size / 2  # Adjust for top face at z=0

        #if hit_normal == Vec3(0, 0, -1):
        #    voxel_center_z += voxel_size

        return Vec3(voxel_center_x, voxel_center_y, voxel_center_z)
    
    @staticmethod
    def get_center_of_hit_dynamic_voxel(raycast_result: BulletClosestHitRayResult) -> Vec3:
        hit_node = raycast_result.getNode()
        hit_object = hit_node.getPythonTag("object")
        ijk = hit_node.getPythonTag("ijk")

        node_np = hit_object.node_paths[ijk]
        return node_np.getPos()
    
    @staticmethod
    def create_voxel_world(chunk_size: int, max_height: int, coordinate: tuple[int, int], voxel_size: float) -> VoxelWorld:
        width = chunk_size
        depth = chunk_size
        
        # Initialize an empty voxel world with air (0)
        world_array = np.zeros((width, depth, max_height), dtype=int)
        
        # Generate or retrieve heightmap for this chunk
        #heightmap = WorldTools.generate_flat_height_map(chunk_size, height=1)
        heightmap = WorldTools.generate_perlin_height_map(chunk_size, coordinate)
        
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

        choices = (
            VoxelType.STONE.value,
            VoxelType.GRASS.value
        )

        world_array = np.where(world_array == 1, np.random.choice(choices, size=world_array.shape), world_array)

        #world_array = np.zeros((5, 5, 5), dtype=int)
        voxel_world = VoxelWorld(world_array, voxel_size)

        '''    
        if coordinate == (0, 0):
            voxel_world.set_voxel(0, 0, 1, VoxelType.GRASS)
        voxel_world.set_voxel(0, 0, 0, VoxelType.STONE)
        voxel_world.set_voxel(1, 0, 0, VoxelType.STONE)
        voxel_world.set_voxel(-1, 0, 0, VoxelType.STONE)
        voxel_world.set_voxel(0, 1, 0, VoxelType.STONE)
        voxel_world.set_voxel(0, -1, 0, VoxelType.STONE)
        voxel_world.set_voxel(1, -1, 0, VoxelType.STONE)
        voxel_world.set_voxel(-1, -1, 0, VoxelType.STONE)
        voxel_world.set_voxel(1, 1, 0, VoxelType.STONE)
        voxel_world.set_voxel(-1, 1, 0, VoxelType.STONE)
        '''

        return voxel_world


    @staticmethod
    def generate_perlin_height_map(chunk_size: int, chunk_coordinate: tuple[int, int]) -> np.ndarray:
        chunk_x, chunk_y = chunk_coordinate
        scale = 0.06  # Adjust scale to control the "zoom" level of the noise
        octaves = 6  # Number of layers of noise to combine
        persistence = 0.5  # Amplitude of each octave
        lacunarity = 1.5  # Frequency of each octave

        height_map = np.zeros((chunk_size + 1, chunk_size + 1))

        # Calculate global offsets
        global_offset_x = chunk_x * chunk_size
        global_offset_y = chunk_y * chunk_size

        for x in range(chunk_size + 1):
            for y in range(chunk_size + 1):
                # Calculate global coordinate
                global_x = (global_offset_x + x) * scale
                global_y = (global_offset_y + y) * scale

                # Generate height using Perlin noise
                height = noise.pnoise2(global_x, global_y,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=10000,  # Large repeat region to avoid repetition
                                    repeaty=10000,
                                    base=0)  # Base can be any constant, adjust for different terrains

                # Map the noise value to a desired height range if needed
                height_map[x, y] = height * 20

        return height_map
    
    @staticmethod
    def generate_flat_height_map(board_size: int, height: int=1) -> np.ndarray:
        # Adjust board_size to account for the extra row and column for seamless edges
        adjusted_size = board_size + 1
        # Create a 2D NumPy array filled with the specified height value
        return np.full((adjusted_size, adjusted_size), height)
    
    @staticmethod
    def calculate_chunk_world_position(coordinate: tuple[int, int], chunk_size: int, voxel_size: float) -> Vec2:
        """
        Calculates the world position of the chunk origo based on its grid coordinate.

        Parameters:
        - coordinate: The chunk's coordinate in the grid/map.
        - chunk_size: Number of voxels along a single axis of the chunk.
        - voxel_size: Size of a voxel

        Returns:
        Vec2: The world position of the chunk.
        """
        chunk_x, chunk_y = coordinate
        x = chunk_x * chunk_size * voxel_size
        y = chunk_y * chunk_size * voxel_size
        return Vec2(x, y)
    
    @staticmethod
    def calculate_world_chunk_coordinate(position: Vec2, chunk_size: int, voxel_size: float) -> tuple[int, int]:
        """
        Calculates the chunk grid coordinate corresponding to a world position.

        Parameters:
        - position: A Vec3 representing the world position.
        - voxel_size: Size of a voxel.
        - chunk_size: Number of voxels along a single axis of the chunk.

        Returns:    
        Tuple[int, int]: The chunk's grid coordinate (chunk_x, chunk_y).
        """
        # Calculate the half-size of a chunk in world units
        half_chunk_size_world_units = (chunk_size * voxel_size) / 2

        # Adjust position to align chunk centers with the origin
        adjusted_pos_x = position.x + half_chunk_size_world_units
        adjusted_pos_y = position.y + half_chunk_size_world_units

        # Calculate chunk coordinate
        chunk_x = math.floor(adjusted_pos_x / (chunk_size * voxel_size)) if adjusted_pos_x >= 0 else math.ceil(adjusted_pos_x / (chunk_size * voxel_size)) - 1
        chunk_y = math.floor(adjusted_pos_y / (chunk_size * voxel_size)) if adjusted_pos_y >= 0 else math.ceil(adjusted_pos_y / (chunk_size * voxel_size)) - 1

        return chunk_x, chunk_y