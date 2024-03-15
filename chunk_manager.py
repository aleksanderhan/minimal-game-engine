import numpy as np
import heapq
import time
from multiprocessing import Pool

from direct.task import Task
from direct.stdpy.threading import Lock

from voxel import VoxelTools
from world import VoxelWorld, WorldTools
from geom import GeometryTools

class ConcurrentSet:

    def __init__(self):
        self.lock = Lock()
        self.set = set()

    def add(self, item):
        with self.lock:
            self.set.add(item)

    def remove(self, item):
        with self.lock:
            self.set.remove(item)

    def __contains__(self, item):
        with self.lock:
            return item in self.set
        
    def __sub__(self, other_set: set):
        return self.set - other_set
    
    def __iter__(self):
        with self.lock:
            # Make a copy for safe iteration
            return iter(list(self.set))


class TaskWrapper:

    def __init__(self, task: tuple[int, int], priority: int):
        self.task = task
        self.priority = priority

class ReprioritizationQueue:

    def __init__(self):
        self.queue: list[tuple[int, int]] = []
        self.lock = Lock()
        self.counter = 0

    def put(self, task: tuple[int, int], priority: int):
        with self.lock:
            self.counter += 1
            wrapper = TaskWrapper(task, priority)
            heapq.heappush(self.queue, (priority, self.counter, wrapper))

    def get(self) -> tuple[int, int]:
        with self.lock:
            if not self.queue:
                return None
            _, _, wrapper = heapq.heappop(self.queue)
            return wrapper.task

    def batch_change_priority(self, wrapped_tasks: list[TaskWrapper]):
        with self.lock:
            # Create a dictionary to map tasks to new priorities
            priority_map = {wrapper.task: wrapper.priority for wrapper in wrapped_tasks}

            # Create a new temporary list for storing updated elements
            new_queue = []

            # Iterate over the existing heap
            for priority, counter, wrapper in self.queue:
                if wrapper.task in priority_map:
                    # Update the priority if necessary
                    new_priority = priority_map[wrapper.task]
                    new_queue.append((new_priority, counter, wrapper))
                else:
                    # Keep the element as is
                    new_queue.append((priority, counter, wrapper))

            # Replace the queue with the modified version
            self.queue = new_queue
            heapq.heapify(self.queue)  # Re-establish heap order


class ChunkManager:
    
    def __init__(self, game_engine):
        self.game_engine = game_engine

        self.loaded_chunks = {}

        self.num_workers = 4
        self.pool = Pool(processes=self.num_workers)
        self.load_queue = ReprioritizationQueue()
        self.unload_queue = ReprioritizationQueue()
        self.chunks_scheduled_for_loading = ConcurrentSet()
        self.chunks_actively_loading = ConcurrentSet()

        self.chunk_radius = 2#8
        self.num_chunks = 2*int(3.14*self.chunk_radius**2)
        self.batch_size = self.num_chunks // self.num_workers
        print("batch_size", self.batch_size)

        #self.game_engine.taskMgr.add(self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")
        #self.game_engine.taskMgr.add(self._unload_chunk_furthest_away, "UnloadFurthestChunks")
        self.game_engine.taskMgr.add(self._load_closest_chunks, "LoadClosestChunks")
        self.game_engine.taskMgr.doMethodLater(2, self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")

    def get_player_chunk_coordinate(self) -> tuple[int, int]:
        player_pos = self.game_engine.camera.getPos()
        return WorldTools.calculate_world_chunk_coordinate(player_pos, self.game_engine.chunk_size, self.game_engine.voxel_size)
    
    def get_player_distance_from_coordinate(self, coordinate: tuple[int, int]) -> float:
        x, y = coordinate
        player_chunk_x, player_chunk_y = self.get_player_chunk_coordinate()
        return ((x - player_chunk_x)**2 + (y - player_chunk_y)**2)**0.5

    def get_voxel_world(self, coordinate: tuple[int, int]) -> VoxelWorld:
        return self.loaded_chunks.get(coordinate)
    
    def get_number_of_loaded_vertices(self) -> int:
        result = 0
        for voxel_world in list(self.loaded_chunks.values()):
            result += len(voxel_world.vertices)
        return result
    
    def get_number_of_visible_voxels(self) -> int:
        result = 0
        for voxel_world in list(self.loaded_chunks.values()):
            exposed_voxels = VoxelTools.identify_exposed_voxels(voxel_world.world_array)
            result += np.count_nonzero(exposed_voxels)
        return result
    
    def load_chunk(self, coordinate: tuple[int, int], voxel_world: VoxelWorld):
        t0 = time.perf_counter()
        self.loaded_chunks[coordinate] = voxel_world
        self.game_engine.apply_texture_and_physics_to_chunk(coordinate, voxel_world)
        dt = time.perf_counter() - t0
        print("load_chunk", dt)
    
    def _callback(self, data: tuple[tuple[int, int], VoxelWorld]):
        coordinate, voxel_world = data
        self.load_chunk(coordinate, voxel_world)
        self.chunks_actively_loading.remove(coordinate)

    def _error_callback(self, exc):
        print('Worker error:', exc)
    
    def _load_closest_chunks(self, task: Task) -> int:
        for _ in range(self.batch_size):
            coordinate = self.load_queue.get()
            if coordinate is not None:
                distance_from_player = self.get_player_distance_from_coordinate(coordinate)
                if distance_from_player <= self.chunk_radius:
                    params = (coordinate, self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_size, self.loaded_chunks, self.game_engine.args.debug)
                    self.chunks_actively_loading.add(coordinate)
                    self.chunks_scheduled_for_loading.remove(coordinate)
                    self.pool.apply_async(ChunkManager._worker, params, callback=self._callback, error_callback=self._error_callback)
        return Task.cont        
    
    @staticmethod
    def _worker(coordinate: tuple[int, int],
                chunk_size: int, 
                max_height: int, 
                voxel_size:int, 
                loaded_chunks: dict, 
                debug: bool) -> tuple[tuple[int, int], VoxelWorld]:
        
        # Generate the chunk and obtain both visual (terrainNP) and physics components (terrainNode)
        t0 = time.perf_counter()
        voxel_world = WorldTools.get_or_create_voxel_world(chunk_size, max_height, loaded_chunks, coordinate, voxel_size)
        voxel_world.create_world_mesh()
        terrain_np = GeometryTools.create_geometry(voxel_world.vertices, voxel_world.indices, debug=debug)
        voxel_world.terrain_np = terrain_np
        dt = time.perf_counter() - t0
        #print("worker", dt)
        return coordinate, voxel_world

    def _identify_chunks_to_load_and_unload(self, task: Task) -> Task:
        print("chunks_actively_loading", [coord for coord in list(self.chunks_actively_loading)])
        print("chunks_scheduled_for_loading", [coord for coord in list(self.chunks_scheduled_for_loading)])
        t0 = time.perf_counter()
        
        handled_tasks = set()
        change_priority = []

        player_chunk_x, player_chunk_y = self.get_player_chunk_coordinate()
        for x in range(player_chunk_x - self.chunk_radius, player_chunk_x + self.chunk_radius + 1):
            for y in range(player_chunk_y - self.chunk_radius, player_chunk_y + self.chunk_radius + 1):
                coordinate = (x, y)
                distance_from_player = self.get_player_distance_from_coordinate(coordinate)
                if distance_from_player <= self.chunk_radius and coordinate not in self.loaded_chunks and coordinate not in self.chunks_scheduled_for_loading:
                    self.chunks_scheduled_for_loading.add(coordinate)
                    self.load_queue.put(coordinate, distance_from_player)  # Direct submission upon identification
                elif distance_from_player > self.chunk_radius and coordinate in self.chunks_scheduled_for_loading:
                    change_priority.append(TaskWrapper(coordinate, distance_from_player))
                handled_tasks.add(coordinate)

        scheduled_for_loading_and_outside_chunk_radius = self.chunks_scheduled_for_loading - handled_tasks
        for coordinate in scheduled_for_loading_and_outside_chunk_radius:
            distance_from_player = self.get_player_distance_from_coordinate(coordinate)
            if distance_from_player > self.chunk_radius:
                self.chunks_scheduled_for_loading.remove(coordinate)
        self.load_queue.batch_change_priority(change_priority)
        dt = time.perf_counter() - t0
        print(dt)
        return Task.again
    
    def _unload_chunk_furthest_away(self, task: Task) -> int:
        chunk = self.unload_queue.get()
        if chunk:
            self._unload_chunk(chunk)
        return Task.cont

    def _unload_chunk(self, coordinate: tuple[int, int]):
        voxel_world = self.loaded_chunks.pop(coordinate, None)
        if voxel_world:
            voxel_world.terrain_np.removeNode()
            self.game_engine.physics_world.removeRigidBody(voxel_world.terrain_node)

