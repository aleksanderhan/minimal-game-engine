import numpy as np
import numba as nb
import heapq
import time
import os
import math
from multiprocessing import Pool, RLock
from typing import Any

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
        self.queue: list[tuple[float, int, TaskWrapper]] = []
        self.scheduled: set[TaskWrapper] = set()
        self.lock = RLock()
        self.counter = 0

    def put(self, task: TaskWrapper):
        with self.lock:
            self.counter += 1
            self.scheduled.add(task)
            heapq.heappush(self.queue, (task.priority, self.counter, task))
    
    def get(self) -> TaskWrapper | None:
        with self.lock:
            if not self.queue:
                return None
            _, _, task = heapq.heappop(self.queue)
            self.scheduled.remove(task)
            return task
        
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

    def batch_remove(self, tasks_to_remove: list[TaskWrapper]):
        with self.lock:
            # Filter out the tasks to remove and rebuild the queue
            self.queue = [(p, c, t) for p, c, t in self.queue if t not in tasks_to_remove]
            for task_to_remove in tasks_to_remove:
                self.scheduled.remove(task_to_remove)
            heapq.heapify(self.queue)


class ChunkManager:
    
    def __init__(self, game_engine):
        self.game_engine = game_engine

        self.loaded_chunks: dict[tuple[int, int], VoxelWorld] = {}

        self.num_workers = math.ceil(os.cpu_count() / 2)
        self.pool = Pool(processes=self.num_workers)
        self.load_queue = ReprioritizationQueue()
        self.unload_queue = ReprioritizationQueue()

        self.chunk_radius = 24
        self.num_chunks = 2*int(3.14*self.chunk_radius**2)
        self.batch_size = self.num_workers * 2
        print("num_chunks", self.num_chunks)
        print("batch_size", self.batch_size)

        #self.game_engine.taskMgr.add(self._load_closest_chunks, "LoadClosestChunks")
        self.game_engine.taskMgr.doMethodLater(0.5, self._load_closest_chunks, "LoadClosestChunks")
        #self.game_engine.taskMgr.doMethodLater(1, self._unload_chunk_furthest_away, "UnloadFurthestChunks")
        self.game_engine.taskMgr.doMethodLater(0.5, self._identify_chunks_to_load_and_unload, "IdentifyChunksToLoadAndUnload")

    def get_player_chunk_coordinates(self) -> tuple[int, int]:
        player_pos = self.game_engine.camera.getPos()
        return WorldTools.calculate_world_chunk_coordinates(player_pos, self.game_engine.chunk_size, self.game_engine.voxel_size)

    def get_voxel_world(self, chunk_coordinates: tuple[int, int]) -> VoxelWorld:
        return self.loaded_chunks.get(chunk_coordinates)
    
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
    
    def load_chunk(self, 
                   coordinates: tuple[int, int],
                   voxel_world: VoxelWorld, 
                   vertices: np.ndarray,
                   indices: np.ndarray):
    
        t0 = time.perf_counter()
        self.game_engine.create_and_apply_mesh_and_physics(coordinates, voxel_world, vertices, indices)
        self.loaded_chunks[coordinates] = voxel_world
        #dt = time.perf_counter() - t0
        #print("load_chunk", voxel_world.chunk_coord, "dt", dt)
    
    def _callback(self, data: tuple[tuple[int, int], VoxelWorld, np.ndarray, np.ndarray]):
        self.load_chunk(*data)

    def _error_callback(self, exc):
        print('Worker error:', exc)
    
    def _load_closest_chunks(self, task: Task) -> int:
        for _ in range(self.batch_size):
            load_task = self.load_queue.get()
            if load_task is not None:
                # Check if the task is already loaded or in process.
                if not load_task in self.load_queue.scheduled and load_task.id not in self.loaded_chunks:
                    params = (load_task.id, self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_size, self.game_engine.args.debug)
                    self.pool.apply_async(ChunkManager._worker, params, callback=self._callback, error_callback=self._error_callback)
        return Task.again

    @staticmethod
    def _worker(coordinates: tuple[int, int],
                chunk_size: int, 
                max_height: int, 
                voxel_size:int, 
                debug: bool) -> tuple[tuple[int, int], VoxelWorld, np.ndarray, np.ndarray]:
        
        # Generate the chunk and obtain both visual (terrain_np) and physics components (terrain_node)
        t0 = time.perf_counter()
        voxel_world = WorldTools.create_voxel_world(chunk_size, max_height, coordinates, voxel_size)
        voxel_world.chunk_coord = coordinates
        vertices, indices = WorldTools.create_world_mesh(voxel_world.world_array, voxel_size)
        voxel_world.terrain_np = GeometryTools.create_geometry(vertices, indices, debug=debug)
        #dt = time.perf_counter() - t0
        return coordinates, voxel_world, vertices, indices
    
    def _identify_chunks_to_load_and_unload(self, task: Task) -> Task:
        chunks_inside_radius: set[TaskWrapper] = set()
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
                        if load_task in self.load_queue.scheduled:
                            change_priority.append(load_task)
                        else:
                            self.load_queue.put(load_task)

        # Remove chunks scheduled for loading that are no longer within the chunk radius of the player
        scheduled_for_loading_and_outside_chunk_radius = self.load_queue.scheduled - chunks_inside_radius
        self.load_queue.batch_remove(scheduled_for_loading_and_outside_chunk_radius)
        
        # Reprioritize load tasks that have changed priority
        self.load_queue.batch_reprioritize_tasks(change_priority)        

        # Add chunks that are furthest away to unload queue
        '''
        chunks_inside_radius_coordinates = set([task.id for task in chunks_inside_radius])
        loaded_and_outside_chunk_radius = set(self.loaded_chunks.keys()) - chunks_inside_radius_coordinates
        for coordinates in loaded_and_outside_chunk_radius:
            distance_from_player = WorldTools.calculate_distance_between_2d_points(coordinates, player_chunk_coords)
            unload_taks = TaskWrapper(coordinates, -distance_from_player)
            self.unload_queue.put(unload_taks)
        '''
        return Task.again
    
    def _unload_chunk_furthest_away(self, task: Task) -> int:
        num_chunks_to_unload = len(self.loaded_chunks) - self.num_chunks
        for _ in range(num_chunks_to_unload):
            unload_tqask = self.unload_queue.get()
            if unload_tqask:
                self._unload_chunk(unload_tqask)
        return Task.again

    def _unload_chunk(self, unload_tqask: TaskWrapper):
        voxel_world = self.loaded_chunks.pop(unload_tqask.id, None)
        if voxel_world:
            voxel_world.terrain_np.removeNode()
            self.game_engine.physics_world.removeRigidBody(voxel_world.terrain_node)

