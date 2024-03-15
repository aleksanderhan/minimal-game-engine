import numpy as np
import heapq
from multiprocessing import Pool

from direct.task import Task
from direct.stdpy.threading import Lock
from panda3d.core import Vec2

from voxel import VoxelTools
from world import VoxelWorld, WorldTools


class ReprioritizationQueue:

    def __init__(self):
        self.queue = []
        self.entry_finder = {}  # Maps tasks to entries
        self.lock = Lock() 
        self.counter = 0

    def put(self, task, priority=0):
        with self.lock:
            self.counter += 1
            if task in self.entry_finder:
                self._remove_task(task)
            entry = [priority, self.counter, task]  
            self.entry_finder[task] = entry
            heapq.heappush(self.queue, entry)

    def get(self):
        with self.lock:
            while not self.queue:  # Check if the queue is empty
                return None        # Exit the get method gracefully 
            priority, count, task = heapq.heappop(self.queue)
            if task is not "REMOVE":
                del self.entry_finder[task]
                return task
            
    def has(self, task):
        return self.entry_finder.get(task) is not None

    def _remove_task(self, task):
        entry = self.entry_finder.pop(task, None)
        if entry:
            # Mark the entry as removed
            entry[2] = "REMOVE"



class ChunkManager:
    
    def __init__(self, game_engine):
        self.game_engine = game_engine

        self.loaded_chunks = {}

        self.num_workers = 6
        self.pool = Pool(processes=self.num_workers)
        self.load_queue = ReprioritizationQueue()
        self.unload_queue = ReprioritizationQueue()
        self.chunks_loading = set()

        self.chunk_radius = 12
        self.num_chunks = int(3.14*self.chunk_radius**2)

        self.game_engine.taskMgr.add(self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")
        self.game_engine.taskMgr.add(self._unload_chunks_furthest_away, "UnloadFurthestChunks")
        self.game_engine.taskMgr.add(self._load_closest_chunks, "LoadClosestChunks")

    def get_voxel_world(self, coordinates: tuple[int, int]) -> VoxelWorld:
        return self.loaded_chunks.get(coordinates)
    
    def get_number_of_loaded_vertices(self) -> int:
        result = 0
        for voxel_world in self.loaded_chunks.values():
            result += len(voxel_world.vertices)
        return result
    
    def get_number_of_visible_voxels(self) -> int:
        result = 0
        for coordinates, world in self.loaded_chunks.items():
            world = self.loaded_chunks.get(coordinates)
            exposed_voxels = VoxelTools.identify_exposed_voxels(world.world_array)
            result += np.count_nonzero(exposed_voxels)
        return result
    
    def load_chunk(self, coordinates: tuple[int, int], voxel_world: VoxelWorld):
        self.loaded_chunks[coordinates] = voxel_world
        self.game_engine.apply_texture_and_physics_to_chunk(coordinates, voxel_world)

    def get_player_chunk_coordinates(self) -> tuple[int, int]:
        player_pos = self.game_engine.camera.getPos()
        return WorldTools.calculate_world_chunk_coordinates(player_pos, self.game_engine.chunk_size, self.game_engine.voxel_size)
    
    def _callback(self, data: tuple[tuple[int, int], VoxelWorld]):
        coordinates, voxel_world = data
        self.load_chunk(coordinates, voxel_world)
        self.chunks_loading.remove(coordinates)

    def _error_callback(self, exc):
        print('Worker error:', exc)
    
    def _load_closest_chunks(self, task: Task) -> int:
        coordinates = self.load_queue.get()
        if coordinates is not None:
            params = (coordinates, self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_size, self.loaded_chunks)
            self.pool.apply_async(ChunkManager._worker, params, callback=self._callback, error_callback=self._error_callback)

        return Task.cont        
    
    @staticmethod
    def _worker(coordinates: tuple[int, int], chunk_size: int, max_height: int, voxel_size:int, loaded_chunks: dict) -> tuple[tuple[int, int], VoxelWorld]:
        # Generate the chunk and obtain both visual (terrainNP) and physics components (terrainNode)
        voxel_world = WorldTools.get_voxel_world(chunk_size, max_height, loaded_chunks, coordinates, voxel_size)
        voxel_world.create_world_mesh()


        '''
            print(f"Generated chunk mesh data in {t1-t0}")
            print(f"    Created world in {create_world_DT}")
            print(f"    Created mesh in {crete_mesh_DT}")
            #print(f"Loaded texture and physics in {t2-t1}")
            #print(f"Unloaded chunks in {t3-t2}")
            print(f"Loaded vertices: {self.get_number_of_loaded_vertices()}")
            print(f"Number of visible voxels: {self.get_number_of_visible_voxels()}")
        '''
        return coordinates, voxel_world

    def _identify_chunks_to_load_and_unload(self, task: Task) -> Task:
        player_chunk_x, player_chunk_y = self.get_player_chunk_coordinates()

        # Iterate through all possible chunk coordinates around the player within the chunk_radius.
        for x in range(player_chunk_x - self.chunk_radius, player_chunk_x + self.chunk_radius + 1):
            for y in range(player_chunk_y - self.chunk_radius, player_chunk_y + self.chunk_radius + 1):
                
                # Calculate the distance from the current chunk to the player's chunk coordinates.
                distance_from_player = ((x - player_chunk_x)**2 + (y - player_chunk_y)**2)**0.5
                
                # Check if the chunk is within the specified radius and not already loaded.
                coordinates = (x, y)
                if distance_from_player <= self.chunk_radius and coordinates not in self.loaded_chunks and coordinates not in self.chunks_loading:
                    # If the chunk meets the criteria, add it to the list of chunks to load.
                    #print("chunk to be loaded", "x, y", x, y, "distance_from_player", distance_from_player)
                    self.chunks_loading.add(coordinates)
                    self.load_queue.put(coordinates, distance_from_player)

        for coordinates in list(self.loaded_chunks.keys()):
            x, y = coordinates
            distance_from_player = ((x - player_chunk_x)**2 + (y - player_chunk_y)**2)**0.5

            if distance_from_player > self.chunk_radius and coordinates not in self.chunks_loading:
                self.unload_queue.put(coordinates, -distance_from_player)

        return Task.cont
    
    def _unload_chunks_furthest_away(self, task: Task) -> int:
        chunk = self.unload_queue.get()
        if chunk:
            self._unload_chunk(chunk)
        return Task.cont

    def _unload_chunk(self, coordinates: tuple[int, int]):
        voxel_world = self.loaded_chunks.pop(coordinates, None)
        if voxel_world:
            voxel_world.terrain_np.removeNode()
            self.game_engine.physics_world.removeRigidBody(voxel_world.terrain_node)



