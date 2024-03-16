import numpy as np
import heapq
import time
import os
import math
from multiprocessing import Pool
from typing import Any, Iterator

from direct.task import Task
from direct.stdpy.threading import Lock

from voxel import VoxelTools
from world import VoxelWorld, WorldTools
from geom import GeometryTools


class ConcurrentSet:

    def __init__(self):
        self.lock = Lock()
        self.set = set()

    def add(self, item: Any):
        with self.lock:
            self.set.add(item)

    def remove(self, item: Any):
        with self.lock:
            self.set.remove(item)

    def __contains__(self, item: Any) -> bool:
        with self.lock:
            return item in self.set
        
    def __sub__(self, other_set: set[Any]) -> set[Any]:
        return self.set - other_set
    
    def __iter__(self) -> Iterator[Any]:
        with self.lock:
            # Make a copy for safe iteration
            return iter(list(self.set))


class TaskWrapper:

    def __init__(self, data: Any, priority: int):
        self.data = data
        self.priority = priority


class ReprioritizationQueue:

    def __init__(self):
        self.queue: list[TaskWrapper] = []
        self.lock = Lock()
        self.counter = 0

    def put(self, task: TaskWrapper):
        with self.lock:
            self.counter += 1
            heapq.heappush(self.queue, (task.priority, self.counter, task))

    def get(self) -> TaskWrapper | None:
        with self.lock:
            if not self.queue:
                return None
            _, _, task = heapq.heappop(self.queue)
            return task.data

    def batch_reprioritize_tasks(self, task_list: list[TaskWrapper]):
        with self.lock:
            # Create a mapping from task to new priority for quick lookup
            new_priorities = {task.data: task.priority for task in task_list}

            # Iterate over the queue and update priorities based on new_priorities
            for i, (_, counter, task) in enumerate(self.queue):
                if task.data in new_priorities:
                    # Update priority directly in the queue
                    self.queue[i] = (new_priorities[task.data], counter, task)

            # After updating the priorities in the queue, re-heapify to maintain the heap invariant
            heapq.heapify(self.queue)

    def remove(self, task_to_remove: TaskWrapper):
        with self.lock:
            # Find the index of the task to remove
            index_to_remove = None
            for i, (_, _, task) in enumerate(self.queue):
                if task.data == task_to_remove.data:
                    index_to_remove = i
                    break

            # If the task was found, remove it
            if index_to_remove is not None:
                self.queue.pop(index_to_remove)
                heapq.heapify(self.queue)


class ChunkManager:
    
    def __init__(self, game_engine):
        self.game_engine = game_engine

        self.loaded_chunks = {}

        self.num_workers = math.ceil(os.cpu_count() / 2)
        self.pool = Pool(processes=self.num_workers)
        self.load_queue = ReprioritizationQueue()
        self.unload_queue = ReprioritizationQueue()
        self.chunks_scheduled_for_loading = ConcurrentSet()
        self.chunks_actively_loading = ConcurrentSet()

        self.chunk_radius = 8
        self.num_chunks = 2*int(3.14*self.chunk_radius**2)
        self.batch_size = self.num_chunks // self.num_workers
        print("batch_size", self.batch_size)

        #self.game_engine.taskMgr.add(self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")
        #self.game_engine.taskMgr.add(self._unload_chunk_furthest_away, "UnloadFurthestChunks")
        self.game_engine.taskMgr.add(self._load_closest_chunks, "LoadClosestChunks")
        self.game_engine.taskMgr.doMethodLater(1, self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")

    def get_player_chunk_coordinate(self) -> tuple[int, int]:
        player_pos = self.game_engine.camera.getPos()
        return WorldTools.calculate_world_chunk_coordinate(player_pos, self.game_engine.chunk_size, self.game_engine.voxel_size)

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
        print("load_chunk", coordinate, "dt", dt)
    
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
                params = (coordinate, self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_size, self.loaded_chunks, self.game_engine.args.debug)
                self.chunks_scheduled_for_loading.remove(coordinate)
                self.chunks_actively_loading.add(coordinate)
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
        t0 = time.perf_counter()
        
        #chunks_scheduled_for_loading = set()
        chunks_inside_radius = set()
        change_priority = []

        # Iterate over a square grid centered on the player to find chunks to load withing the chunk radius, centered on the player. 
        player_chunk_coords = self.get_player_chunk_coordinate()
        player_chunk_x, player_chunk_y = player_chunk_coords
        for x in range(player_chunk_x - self.chunk_radius, player_chunk_x + self.chunk_radius + 1):
            for y in range(player_chunk_y - self.chunk_radius, player_chunk_y + self.chunk_radius + 1):
                coordinate = (x, y)
                distance_from_player = WorldTools.calculate_distance_between_2d_points(coordinate, player_chunk_coords)
                if distance_from_player < self.chunk_radius: 
                    chunks_inside_radius.add(coordinate) # Add the chunks outside the radius, but inside the square
                    if coordinate not in self.loaded_chunks:
                        if coordinate not in self.chunks_scheduled_for_loading:
                            self.chunks_scheduled_for_loading.add(coordinate)
                            self.load_queue.put(TaskWrapper(coordinate, distance_from_player))
                    if coordinate in self.chunks_scheduled_for_loading:
                        # TODO check if priority should be updated
                        change_priority.append(TaskWrapper(coordinate, distance_from_player))

        scheduled_for_loading_and_outside_chunk_radius = self.chunks_scheduled_for_loading - chunks_inside_radius
        for coordinate in scheduled_for_loading_and_outside_chunk_radius:
            self.load_queue.remove(TaskWrapper(coordinate, None))
            self.chunks_scheduled_for_loading.remove(coordinate)

        self.load_queue.batch_reprioritize_tasks(change_priority)

        dt = time.perf_counter() - t0
        print("_identify_chunks_to_load_and_unload dt", dt)
        
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

