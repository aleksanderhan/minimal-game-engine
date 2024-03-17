import numpy as np
import noise
import math
from functools import lru_cache

from panda3d.core import Vec3, Vec2, NodePath
from panda3d.bullet import BulletClosestHitRayResult, BulletRigidBodyNode

from constants import VoxelType, voxel_type_map
from utils import voxel_grid_coordinates_to_index


class VoxelWorld:

    def __init__(self, world_array: np.ndarray, voxel_size: float):

        self.voxel_size = voxel_size
        self.world_array = world_array
        self.chunk_coord: tuple[int, int] = None

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
        i, j, k = voxel_grid_coordinates_to_index(ix, iy, iz, self.world_array.shape[0])
        voxel_type_int = self.world_array[i, j, k]
        return voxel_type_map[voxel_type_int]

    def set_voxel(self, ix: int, iy: int, iz: int, voxel_type: VoxelType):
        i, j, k = voxel_grid_coordinates_to_index(ix, iy, iz, self.world_array.shape[0])
        self.world_array[i, j, k] = voxel_type.value


def calculate_distance_between_2d_points(point1: tuple[int, int] | Vec2, point2: tuple[int, int] | Vec2) -> float:
    point1_x, point1_y = point1
    point2_x, point2_y = point2
    return ((point2_x - point1_x)**2 + (point2_y - point1_y)**2)**0.5

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

    # Convert the adjusted hit position to voxel grid coordinates
    grid_x = round(adjusted_hit_pos.x / voxel_size)
    grid_y = round(adjusted_hit_pos.y / voxel_size)
    grid_z = round(adjusted_hit_pos.z / voxel_size)

    # Calculate the center of the voxel in world coordinates
    voxel_center_x = grid_x * voxel_size
    voxel_center_y = grid_y * voxel_size
    voxel_center_z = grid_z * voxel_size - voxel_size / 2  # Adjust for top face at z=0

    #if hit_normal == Vec3(0, 0, -1):
    #    voxel_center_z += voxel_size

    return Vec3(voxel_center_x, voxel_center_y, voxel_center_z)

def get_center_of_hit_dynamic_voxel(raycast_result: BulletClosestHitRayResult) -> Vec3:
    hit_node = raycast_result.getNode()
    hit_object = hit_node.getPythonTag("object")
    ijk = hit_node.getPythonTag("ijk")

    node_np = hit_object.node_paths[ijk]
    return node_np.getPos()

def create_voxel_world(chunk_size: int, max_height: int, chunk_coordinates: tuple[int, int], voxel_size: float) -> VoxelWorld:
    width = chunk_size
    depth = chunk_size
    
    # Initialize an empty voxel world with air (0)
    world_array = np.zeros((width, depth, max_height), dtype=int)
    
    # Generate or retrieve heightmap for this chunk
    #heightmap = generate_flat_height_map(chunk_size, height=1)
    heightmap = generate_perlin_height_map(chunk_size, chunk_coordinates)
    
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
    if coordinates == (0, 0):
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

def generate_perlin_height_map(chunk_size: int, chunk_coordinates: tuple[int, int]) -> np.ndarray:
    chunk_x, chunk_y = chunk_coordinates
    scale = 0.005  # Adjust scale to control the "zoom" level of the noise
    octaves = 4  # Number of layers of noise to combine
    persistence = 2.5  # Amplitude of each octave
    lacunarity = 1.5  # Frequency of each octave

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
                                base=0)  # Base can be any constant, adjust for different terrains

            # Map the noise value to a desired height range if needed
            height_map[x, y] = height * 70

    return height_map

@lru_cache
def generate_flat_height_map(board_size: int, height: int=1) -> np.ndarray:
    # Adjust board_size to account for the extra row and column for seamless edges
    adjusted_size = board_size + 1
    # Create a 2D NumPy array filled with the specified height value
    return np.full((adjusted_size, adjusted_size), height)

def calculate_chunk_world_position(chunk_coordinates: tuple[int, int], chunk_size: int, voxel_size: float) -> Vec2:
    """
    Calculates the world position of the chunk origo based on its grid coordinates.

    Parameters:
    - coordinates: The chunk's coordinates in the grid/map.
    - chunk_size: Number of voxels along a single axis of the chunk.
    - voxel_size: Size of a voxel

    Returns:
    Vec2: The world position of the chunk.
    """
    chunk_x, chunk_y = chunk_coordinates
    x = chunk_x * chunk_size * voxel_size
    y = chunk_y * chunk_size * voxel_size
    return Vec2(x, y)

def calculate_world_chunk_coordinates(position: Vec2, chunk_size: int, voxel_size: float) -> tuple[int, int]:
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