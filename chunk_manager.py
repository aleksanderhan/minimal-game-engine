import numpy as np
import heapq
import time
import os
import math
from multiprocessing import Pool, RLock
from typing import Any, Iterator

from direct.task import Task

from voxel import VoxelTools
from world import VoxelWorld, WorldTools
from geom import GeometryTools


class TaskWrapper:

    def __init__(self, id: Any, priority: int):
        self.id = id
        self.priority = priority

    def __eq__(self, other) -> bool:
        return self.id == other.id # Do not add: and self.priority == other.priority - because we use the method for checking if a task is in queue
    
    def __hash__(self) -> int:
        return hash((self.id))


class ReprioritizationQueue:

    def __init__(self):
        self.queue: list[TaskWrapper] = []
        self.scheduled_for_loading: set[TaskWrapper] = set()
        self.lock = RLock()
        self.counter = 0

    def put(self, task: TaskWrapper):
        with self.lock:
            self.counter += 1
            heapq.heappush(self.queue, (task.priority, self.counter, task))
            self.scheduled_for_loading.add(task)
    
    def get(self) -> TaskWrapper | None:
        with self.lock:
            if not self.queue:
                return None
            _, _, task = heapq.heappop(self.queue)
            self.scheduled_for_loading.remove(task)
            return task
    
    def task_in_queue(self, task: TaskWrapper) -> bool:
        return task in self.scheduled_for_loading

    def batch_reprioritize_tasks(self, task_list: list[TaskWrapper]):
        with self.lock:
            # Create a mapping from task to new priority for quick lookup
            new_priorities = {task.id: task.priority for task in task_list}

            # Iterate over the queue and update priorities based on new_priorities
            for i, (_, counter, task) in enumerate(self.queue):
                if task.id in new_priorities:
                    # Update priority directly in the queue
                    self.queue[i] = (new_priorities[task.id], counter, task)

            # After updating the priorities in the queue, re-heapify to maintain the heap invariant
            heapq.heapify(self.queue)

    def remove(self, task_id_to_remove: tuple[int, int]):
        with self.lock:
            # Find the index of the task to remove
            index_to_remove = None
            for i, (_, _, task) in enumerate(self.queue):
                if task.id == task_id_to_remove:
                    index_to_remove = i
                    break

            # If the task was found, remove it
            if index_to_remove is not None:
                self.queue.pop(index_to_remove)
                heapq.heapify(self.queue)
                self.scheduled_for_loading.remove(task)


class ChunkManager:
    
    def __init__(self, game_engine):
        self.game_engine = game_engine

        self.loaded_chunks: dict[tuple[int, int], VoxelWorld] = {}

        self.num_workers = math.ceil(os.cpu_count() / 2)
        self.pool = Pool(processes=self.num_workers)
        self.load_queue = ReprioritizationQueue()
        self.unload_queue = ReprioritizationQueue()

        self.chunk_radius = 8
        self.num_chunks = 4*int(3.14*self.chunk_radius**2)
        self.batch_size = self.num_workers * 2
        print("num_chunks", self.num_chunks)
        print("batch_size", self.batch_size)

        self.game_engine.taskMgr.add(self._unload_chunk_furthest_away, "UnloadFurthestChunks")
        self.game_engine.taskMgr.add(self._load_closest_chunks, "LoadClosestChunks")
        self.game_engine.taskMgr.doMethodLater(1, self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")

    def get_player_chunk_coordinates(self) -> tuple[int, int]:
        player_pos = self.game_engine.camera.getPos()
        return WorldTools.calculate_world_chunk_coordinates(player_pos, self.game_engine.chunk_size, self.game_engine.voxel_size)

    def get_voxel_world(self, coordinates: tuple[int, int]) -> VoxelWorld:
        return self.loaded_chunks.get(coordinates)
    
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
    
    def load_chunk(self, voxel_world: VoxelWorld):
        t0 = time.perf_counter()
        self.loaded_chunks[voxel_world.chunk_coord] = voxel_world
        self.game_engine.apply_texture_and_physics_to_chunk(voxel_world.chunk_coord, voxel_world)
        dt = time.perf_counter() - t0
        #print("load_chunk", voxel_world.chunk_coord, "dt", dt)
    
    def _callback(self, id: tuple[tuple[int, int], VoxelWorld]):
        coordinates, voxel_world = id
        self.loaded_chunks[coordinates] = voxel_world
        self.load_chunk(voxel_world)

    def _error_callback(self, exc):
        print('Worker error:', exc)
    
    def _load_closest_chunks(self, task: Task) -> int:
        for _ in range(self.batch_size):
            load_task = self.load_queue.get()
            if load_task is not None and not load_task in self.load_queue.scheduled_for_loading:
                params = (load_task.id, self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_size, self.game_engine.args.debug)
                self.pool.apply_async(ChunkManager._worker, params, callback=self._callback, error_callback=self._error_callback)
        return Task.cont
    
    @staticmethod
    def _worker(coordinates: tuple[int, int],
                chunk_size: int, 
                max_height: int, 
                voxel_size:int, 
                debug: bool) -> tuple[tuple[int, int], VoxelWorld]:
        
        # Generate the chunk and obtain both visual (terrain_np) and physics components (terrain_node)
        t0 = time.perf_counter()
        voxel_world = WorldTools.create_voxel_world(chunk_size, max_height, coordinates, voxel_size)
        voxel_world.chunk_coord = coordinates
        voxel_world.create_world_mesh()
        terrain_np = GeometryTools.create_geometry(voxel_world.vertices, voxel_world.indices, debug=debug)
        voxel_world.terrain_np = terrain_np
        dt = time.perf_counter() - t0
        #print("worker", dt)
        return coordinates, voxel_world
    
    def _identify_chunks_to_load_and_unload(self, task: Task) -> Task:
        t0 = time.perf_counter()
        
        chunks_inside_radius = set()
        change_priority: list[TaskWrapper] = []

        # Iterate over a square grid centered on the player to find chunks to load withing the chunk radius, centered on the player. 
        player_chunk_coords = self.get_player_chunk_coordinates()
        player_chunk_x, player_chunk_y = player_chunk_coords
        for x in range(player_chunk_x - self.chunk_radius, player_chunk_x + self.chunk_radius + 1):
            for y in range(player_chunk_y - self.chunk_radius, player_chunk_y + self.chunk_radius + 1):
                coordinates = (x, y)
                
                distance_from_player = WorldTools.calculate_distance_between_2d_points(coordinates, player_chunk_coords)
                if distance_from_player < self.chunk_radius: 
                    load_task = TaskWrapper(coordinates, distance_from_player)
                    chunks_inside_radius.add(load_task) # Add the chunks outside the radius, but inside the square
                    
                    if coordinates not in self.loaded_chunks:
                        if load_task in self.load_queue.scheduled_for_loading:
                            change_priority.append(load_task)
                        else:
                            self.load_queue.put(load_task)

        scheduled_for_loading_and_outside_chunk_radius = self.load_queue.scheduled_for_loading - chunks_inside_radius
        for load_task in scheduled_for_loading_and_outside_chunk_radius:
            self.load_queue.remove(load_task.id)

        self.load_queue.batch_reprioritize_tasks(change_priority)

        dt = time.perf_counter() - t0
        #print("_identify_chunks_to_load_and_unload dt", dt)
        
        return Task.again
    
    def _unload_chunk_furthest_away(self, task: Task) -> int:
        chunk = self.unload_queue.get()
        if chunk:
            self._unload_chunk(chunk)
        return Task.cont

    def _unload_chunk(self, coordinates: tuple[int, int]):
        voxel_world = self.loaded_chunks.pop(coordinates, None)
        if voxel_world:
            voxel_world.terrain_np.removeNode()
            self.game_engine.physics_world.removeRigidBody(voxel_world.terrain_node)

