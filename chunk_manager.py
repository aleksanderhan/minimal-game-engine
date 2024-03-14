import time
import numpy as np
from multiprocessing import Pool
from direct.stdpy import threading

from helper import VoxelTools, WorldTools
from world import VoxelWorld

class ChunkManager:
    
    def __init__(self, game_engine):
        self.voxel_world_map = {}
        self.game_engine = game_engine
        self.loaded_chunks = {}
        self.pool = Pool(processes=6)
        self.previously_updated_position = None  # Initialize with None or with the player's starting position
        self.inner_radius = 8
        self.chunk_radius = 12
        self.num_chunks = 4*int(3.14*self.chunk_radius**2)

    def get_voxel_world(self, chunk_x: int, chunk_y: int) -> VoxelWorld:
        return self.voxel_world_map.get((chunk_x, chunk_y)) 

    def get_player_chunk_pos(self):
        player_pos = self.game_engine.camera.getPos()
        chunk_x, chunk_y = WorldTools.calculate_world_chunk_position(player_pos, self.game_engine.chunk_size, self.game_engine.voxel_size)
        return chunk_x, chunk_y

    def update_chunks(self):
        T0 = time.perf_counter()
        player_chunk_x, player_chunk_y = self.get_player_chunk_pos()

        if self.previously_updated_position: 
            distance_from_center = ((player_chunk_x - self.previously_updated_position[0])**2 + 
                                    (player_chunk_y - self.previously_updated_position[1])**2)**0.5
            if distance_from_center <= self.inner_radius:
                return  # Player still within inner radius, no loading needed
            
        chunks_to_load = self.identify_chunks_to_load(player_chunk_x, player_chunk_y, self.chunk_radius)
        
        # Use multiprocessing to generate chunks
        t0 = time.perf_counter()
        chunk_data = self.pool.starmap(WorldTools.generate_chunk,
                        [(self.game_engine.chunk_size, self.game_engine.max_height, self.voxel_world_map, x, y, self.game_engine.voxel_size) for x, y in chunks_to_load])

        # Apply textures and physics sequentially
        t1 = time.perf_counter()

        create_world_DT = 0
        crete_mesh_DT = 0
        for (x, y), (vertices, indices, voxel_world, create_world_dt, crete_mesh_dt) in zip(chunks_to_load, chunk_data):
            create_world_DT += create_world_dt
            crete_mesh_DT += crete_mesh_dt
            self.voxel_world_map[(x, y)] = voxel_world
            terrainNP, terrainNode = self.game_engine.apply_texture_and_physics_to_chunk(x, y, vertices, indices)
            self.loaded_chunks[(x, y)] = (terrainNP, terrainNode, len(vertices))

        # Update previously_updated_position with the current player position after loading chunks
        self.previously_updated_position = (player_chunk_x, player_chunk_y)

        t2 = time.perf_counter()
        # Unload chunks outside the new radius
        self.unload_chunks_furthest_away(player_chunk_x, player_chunk_y, self.chunk_radius)
        t3 = time.perf_counter()

        if self.game_engine.args.debug:
            print(f"Generated chunk mesh data in {t1-t0}")
            print(f"    Created world in {create_world_DT}")
            print(f"    Created mesh in {crete_mesh_DT}")
            print(f"Loaded texture and physics in {t2-t1}")
            print(f"Unloaded chunks in {t3-t2}")
            print(f"Loaded vertices: {self.get_number_of_loaded_vertices()}")
            print(f"Number of visible voxels: {self.get_number_of_visible_voxels()}")

        DT = time.perf_counter() - T0
        print(f"Loaded chunks in {DT}")
        print()

    def identify_chunks_to_load(self, player_chunk_x, player_chunk_y, chunk_radius):
        # Initialize an empty list to store the coordinates of chunks that need to be loaded.
        chunks_to_load = []

        # Iterate through all possible chunk coordinates around the player within the chunk_radius.
        for x in range(player_chunk_x - chunk_radius, player_chunk_x + chunk_radius + 1):
            for y in range(player_chunk_y - chunk_radius, player_chunk_y + chunk_radius + 1):
                
                # Calculate the distance from the current chunk to the player's chunk position.
                distance_from_player = ((x - player_chunk_x)**2 + (y - player_chunk_y)**2)**0.5
                
                # Check if the chunk is within the specified radius and not already loaded.
                if distance_from_player <= chunk_radius and (x, y) not in self.loaded_chunks:
                    # If the chunk meets the criteria, add it to the list of chunks to load.
                    if (x, y) not in self.loaded_chunks:
                        chunks_to_load.append((x, y))

        # Return the list of chunks that need to be loaded.
        return chunks_to_load

    def unload_chunks_furthest_away(self, player_chunk_x, player_chunk_y, chunk_radius):
        # Calculate distance for each loaded chunk and keep track of their positions and distances
        chunk_distances = [
            (chunk_pos, ((chunk_pos[0] - player_chunk_x)**2 + (chunk_pos[1] - player_chunk_y)**2))
            for chunk_pos in self.loaded_chunks.keys()
        ]
        
        # Sort chunks by their distance in descending order (furthest first)
        chunk_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Select the furthest self.num_chunks to unload
        chunks_to_unload = list(filter(lambda x: x[1] > chunk_radius, chunk_distances[:len(self.loaded_chunks) - self.num_chunks]))
        
        # Unload these chunks
        for chunk_pos, _ in chunks_to_unload:
            self.unload_chunk(*chunk_pos)
            
        print(f"Unloaded {len(chunks_to_unload)} furthest chunks.")
        
    def get_number_of_loaded_vertices(self):
        result = 0
        for _, _, num_vertices in self.loaded_chunks.values():
            result += num_vertices
        return result
    
    def get_number_of_visible_voxels(self):
        result = 0
        for key in self.loaded_chunks.keys():
            world = self.voxel_world_map.get(key)
            exposed_voxels = VoxelTools.identify_exposed_voxels(world.world_array)
            result += np.count_nonzero(exposed_voxels)
        return result

    def load_chunk(self, chunk_x, chunk_y):
        # Generate the chunk and obtain both visual (terrainNP) and physics components (terrainNode)
        vertices, indices, _, _, _ = WorldTools.generate_chunk(self.game_engine.chunk_size, self.game_engine.max_height, self.voxel_world_map, chunk_x, chunk_y, self.game_engine.voxel_size)
        terrainNP, terrainNode = self.game_engine.apply_texture_and_physics_to_chunk(chunk_x, chunk_y, vertices, indices)
        # Store both components in the loaded_chunks dictionary
        self.loaded_chunks[(chunk_x, chunk_y)] = (terrainNP, terrainNode, len(vertices)) # TODO: Use Chunk dataclass

    def unload_chunk(self, chunk_x, chunk_y):
        chunk_data = self.loaded_chunks.pop((chunk_x, chunk_y), None)
        if chunk_data:
            terrainNP, terrainNode, _ = chunk_data
            terrainNP.removeNode()
            self.game_engine.physics_world.removeRigidBody(terrainNode)