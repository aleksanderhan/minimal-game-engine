import numpy as np
import numba as nb
import time
import heapq
from direct.stdpy.threading import RLock
from typing import Any


def identify_exposed_voxels(voxel_world):
    # Pad the voxel world with zeros (air) on all sides
    padded_world = np.pad(voxel_world, pad_width=1, mode='constant', constant_values=0)
    
    # Create shifted versions of the world for all six directions
    shifts = {
        'left':  (0, -1, 0),
        'right': (0, 1, 0),
        'down':  (-1, 0, 0),
        'up':    (1, 0, 0),
        'back':  (0, 0, -1),
        'front': (0, 0, 1),
    }
    exposed_faces = np.zeros_like(voxel_world, dtype=bool)
    
    for direction, (dx, dy, dz) in shifts.items():
        shifted_world = np.roll(padded_world, shift=(dx, dy, dz), axis=(0, 1, 2))
        # Expose face if there's air next to it (voxel value of 0 in the shifted world)
        exposed_faces |= ((shifted_world[1:-1, 1:-1, 1:-1] == 0) & (voxel_world > 0))
    
    return exposed_faces

def check_surrounding_air_vectorized(voxel_world, x, y, z):
    """
    Vectorized version to check each of the six directions around a point (x, y, z) in the voxel world
    for air (assumed to be represented by 0).
    """
    # Pad the voxel world with 1 (solid) to handle edge cases without manual boundary checks
    padded_world = np.pad(voxel_world, pad_width=1, mode='constant', constant_values=0)

    # Adjust coordinates due to padding
    x, y, z = x + 1, y + 1, z + 1

    # Check the six directions around the point for air, vectorized
    left = padded_world[x - 1, y, z] == 0
    right = padded_world[x + 1, y, z] == 0
    down = padded_world[x, y - 1, z] == 0
    up = padded_world[x, y + 1, z] == 0
    front = padded_world[x, y, z - 1] == 0
    back = padded_world[x, y, z + 1] == 0

    faces = ["left", "right", "down", "up", "back", "front"]
    exposed = [left, right, down, up, back, front]
    
    return [face for face, exp in zip(faces, exposed) if exp]

voxel_world1 = np.zeros((3,3,3), dtype=int)
voxel_world1[0, 0, 0] = 1
voxel_world1[0, 1, 0] = 1

voxel_world2 = np.zeros((3,3,3), dtype=int)
voxel_world2[1, 1, 0] = 1
voxel_world2[1, 1, 1] = 1


voxels1 = identify_exposed_voxels(voxel_world1)
voxels2 = identify_exposed_voxels(voxel_world2)

#print("identify_exposed_voxels1:", voxels1)
#print("identify_exposed_voxels2:", voxels2)


#print("check_surrounding_air_vectorized1:", check_surrounding_air_vectorized(voxel_world1, 1, 1, 1))
#print("check_surrounding_air_vectorized2:", check_surrounding_air_vectorized(voxel_world2, 1, 1, 1))


# Defining a mock class to simulate the VoxelWorld behavior for unit testing
class MockVoxelWorld:
    def __init__(self, array):
        self.array = array
    
    def shape(self):
        return self.array.shape
    
    def get_voxel(self, x, y, z):
        return self.array[x, y, z]

# Re-defining the check_surrounding_air function to use the MockVoxelWorld
def check_surrounding_air(voxel_world: MockVoxelWorld, x, y, z):
    max_x, max_y, max_z = voxel_world.shape()[0] - 1, voxel_world.shape()[1] - 1, voxel_world.shape()[2] - 1
    exposed_faces = []
    
    if x == max_x or voxel_world.get_voxel(x + 1, y, z) == 0: exposed_faces.append("right")
    if x == 0 or voxel_world.get_voxel(x - 1, y, z) == 0: exposed_faces.append("left")
    if y == max_y or voxel_world.get_voxel(x, y + 1, z) == 0: exposed_faces.append("front")
    if y == 0 or voxel_world.get_voxel(x, y - 1, z) == 0: exposed_faces.append("back")
    if z == max_z or voxel_world.get_voxel(x, y, z + 1) == 0: exposed_faces.append("up")
    if z == 0 or voxel_world.get_voxel(x, y, z - 1) == 0: exposed_faces.append("down")
    
    return exposed_faces

# Writing a unit test
def test_check_surrounding_air():
    import numpy as np
    voxel_array = np.array([[[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 1, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]])
    voxel_world = MockVoxelWorld(voxel_array)
    result = check_surrounding_air(voxel_world, 3, 3, 2)
    expected = ["right", "left", "front", "back", "up", "down"]
    print("result", result)
    #assert set(result) == set(expected), f"Expected {expected}, but got {result}"

# Running the unit test
#test_check_surrounding_air()



def extend_array(voxel_array, new_index):
    shape = voxel_array.shape
    
    # Determine the padding needed for each dimension
    pad_width = [(0, 0)] * 3  # Initialize with no padding
    
    for dim in range(3):
        if new_index[dim] < 0:
            pad_width[dim] = (-new_index[dim], 0)
        elif new_index[dim] >= shape[dim]:
            pad_width[dim] = (0, new_index[dim] - shape[dim] + 1)
            
    # Pad the array
    return np.pad(voxel_array, pad_width=pad_width, mode='constant', constant_values=0)

def extend_array_uniformly(voxel_array):
    # Specify the padding width of 1 for all sides of all dimensions
    pad_width = [(1, 1)] * 3  # Padding for depth, rows, and columns
    
    # Pad the array with 0's on all sides
    return np.pad(voxel_array, pad_width=pad_width, mode='constant', constant_values=0)

#a = np.ones((1, 1, 1), dtype=int)
#print(a)
#pa = extend_array_uniformly(a)
#print(pa)



normals = {
    'front':  ( 0,  1,  0),
    'back':   (0,  -1,  0),
    'right':  ( 1,  0,  0),
    'left':   ( -1, 0,  0),
    'up':   ( 0,  0, 1),
    'down':     ( 0,  0,  -1),
}


def identify_exposed_voxels0(world_array: np.ndarray) -> np.ndarray:
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

def identify_exposed_voxels1(world_array: np.ndarray) -> np.ndarray:
    """
    Identifies a voxel exposed to air and returns a same shaped boolean np array with the result.
    True means it is exposed to air, False means it's not.

    Parameters:
        - world_array: a 3D numpy array representing the voxel types as integers in the world
    """
    # Pad the voxel world with zeros (air) on all sides
    padded_world = np.pad(world_array, pad_width=1, mode='constant', constant_values=0)
    
    # Initialize a boolean array for exposed faces
    exposed_faces = np.zeros_like(world_array, dtype=bool)
    
    # Check all six directions in a vectorized manner
    # Left (-x)
    exposed_faces |= ((padded_world[:-2, 1:-1, 1:-1] == 0) & (world_array > 0))
    # Right (+x)
    exposed_faces |= ((padded_world[2:, 1:-1, 1:-1] == 0) & (world_array > 0))
    # Down (-y)
    exposed_faces |= ((padded_world[1:-1, :-2, 1:-1] == 0) & (world_array > 0))
    # Up (+y)
    exposed_faces |= ((padded_world[1:-1, 2:, 1:-1] == 0) & (world_array > 0))
    # Back (-z)
    exposed_faces |= ((padded_world[1:-1, 1:-1, :-2] == 0) & (world_array > 0))
    # Front (+z)
    exposed_faces |= ((padded_world[1:-1, 1:-1, 2:] == 0) & (world_array > 0))
    
    return exposed_faces



world_array = np.random.randint(0, 2, size=(500, 500, 500))

t0 = time.perf_counter()
res1 = identify_exposed_voxels0(world_array)
t1 = time.perf_counter()
res2 = identify_exposed_voxels1(world_array)
t2 = time.perf_counter()
print("identify_exposed_voxels0", t1 - t0)
print("identify_exposed_voxels1", t2 - t1)
assert np.array_equal(res1, res2)
print()



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
        self.scheduled: set[TaskWrapper] = set()
        self.lock = RLock()
        self.counter = 0

    def put(self, task: TaskWrapper):
        print("put", task.id)
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
    
    def task_in_queue(self, task: TaskWrapper) -> bool:
        return task in self.scheduled

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
                        

def calculate_distance_between_2d_points(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    point1_x, point1_y = point1
    point2_x, point2_y = point2
    return ((point2_x - point1_x)**2 + (point2_y - point1_y)**2)**0.5

def _identify_chunks_to_load_and_unload(load_queue, loaded_chunks, player_chunk_coords):
    chunks_inside_radius: set[TaskWrapper] = set()
    change_priority: list[TaskWrapper] = []

    # Iterate over a square grid centered on the player to find chunks to load withing the chunk radius, centered on the player.
    chunk_radius = 2
    player_chunk_x, player_chunk_y = player_chunk_coords
    for x in range(player_chunk_x - chunk_radius, player_chunk_x + chunk_radius + 1):
        for y in range(player_chunk_y - chunk_radius, player_chunk_y + chunk_radius + 1):
            coordinates = (x, y)
            
            distance_from_player = calculate_distance_between_2d_points(coordinates, player_chunk_coords)
            if distance_from_player < chunk_radius: 
                load_task = TaskWrapper(coordinates, distance_from_player)
                chunks_inside_radius.add(load_task) # Add the chunks outside the radius, but inside the square
                
                if coordinates not in loaded_chunks:
                    if load_task in load_queue.scheduled:
                        change_priority.append(load_task)
                    else:
                        load_queue.put(load_task)

    # Remove chunks scheduled for loading that are no longer within the chunk radius of the player
    scheduled_for_loading_and_outside_chunk_radius = load_queue.scheduled - chunks_inside_radius
    print("scheduled_for_loading_and_outside_chunk_radius", [task.id for task in scheduled_for_loading_and_outside_chunk_radius])
    print("chunks_inside_radius", [task.id for task in chunks_inside_radius])

    
    load_queue.batch_reprioritize_tasks(change_priority)
    load_queue.batch_remove(scheduled_for_loading_and_outside_chunk_radius)
    # Reprioritize load tasks that have changed priority





queue = ReprioritizationQueue()

origin = TaskWrapper((0, 0), 0)
front = TaskWrapper((0, 1), 1)
left = TaskWrapper((-1, 0), 1)
right = TaskWrapper((1, 0), 1)
back = TaskWrapper((0, -1), 1)

frontright = TaskWrapper((1, 1), 2**0.5)
frontleft = TaskWrapper((1, -1), 2**0.5)
backright = TaskWrapper((-1, 1), 2**0.5)
backleft = TaskWrapper((-1, -1), 2**0.5)

frontfront = TaskWrapper((0, 2), 2)
leftleft = TaskWrapper((-2, 0), 2)
rightright = TaskWrapper((2, 0), 2)
backback = TaskWrapper((0, -2), 2)


loaded_chunks: dict[tuple[int, int], str] = {}
loaded_chunks[origin.id] = "origin"

loaded_chunks[frontfront.id] = "frontfront"
loaded_chunks[leftleft.id] = "leftleft"
loaded_chunks[rightright.id] = "rightright"
loaded_chunks[backback.id] = "backback"



#queue.put(origin)
queue.put(front)
queue.put(left)
queue.put(right)
queue.put(back)

queue.put(frontright)
queue.put(frontleft)
queue.put(backright)
queue.put(backleft)

queue.put(frontfront)
queue.put(leftleft)
queue.put(rightright)
queue.put(backback)



print("queue.scheduled before", [task.id for task in queue.scheduled])
print("queue.queue before", [task.id for _, _, task in queue.queue])

player_chunk_coords = (0, 0)
_identify_chunks_to_load_and_unload(queue, loaded_chunks, player_chunk_coords)

print("queue.scheduled after", [task.id for task in queue.scheduled])
print("queue.queue after", [task.id for _, _, task in queue.queue])

player_chunk_coords = (1, 0)
_identify_chunks_to_load_and_unload(queue, loaded_chunks, player_chunk_coords)

print("queue.scheduled end", [task.id for task in queue.scheduled])
print("queue.queue end", [task.id for _, _, task in queue.queue])




def identify_exposed_voxels0(voxel_array: np.ndarray) -> np.ndarray:
    """
    Identifies a voxel exposed to air and returns a same shaped boolean np array with the result.
    True means it is exposed to air, False means it's not.

    Parameters:
        - world_array: a 3D numpy array representing the voxel types as integers in the world
    """
    # Pad the voxel world with zeros (air) on all sides
    padded_world = np.pad(voxel_array, pad_width=1, mode='constant', constant_values=0)
    
    # Initialize a boolean array for exposed faces
    exposed_faces = np.zeros_like(voxel_array, dtype=bool)
    
    # Check all six directions in a vectorized manner
    exposed_faces |= ((padded_world[:-2, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[2:, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, :-2, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 2:, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, :-2] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, 2:] == 0) & (voxel_array > 0))
    
    return exposed_faces

import numba as nb

@nb.jit(nopython=True, cache=True)
def identify_exposed_voxels(voxel_array: np.ndarray) -> np.ndarray:
    """
    Identifies voxels exposed to air and returns a boolean array of the same shape as `voxel_array`
    indicating exposure. True means the voxel is exposed to air, False means it's not.

    Parameters:
        - voxel_array: a 3D numpy array representing voxel types as integers in the world.
    """
    # Create a new array with padding of 1 around the original array
    padded_shape = (voxel_array.shape[0] + 2, voxel_array.shape[1] + 2, voxel_array.shape[2] + 2)
    padded_world = np.zeros(padded_shape, dtype=voxel_array.dtype)
    
    # Fill the inner part of the padded array with the original voxel data
    padded_world[1:-1, 1:-1, 1:-1] = voxel_array
    
    # Initialize a boolean array for exposed faces
    exposed_faces = np.zeros_like(voxel_array, dtype=np.bool_)

    # Check all six directions
    exposed_faces |= ((padded_world[:-2, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[2:, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, :-2, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 2:, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, :-2] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, 2:] == 0) & (voxel_array > 0))
    
    return exposed_faces




random_array = np.random.randint(0, 1, size=(100, 100, 100))

print()

t0 = time.perf_counter()
identify_exposed_voxels0(random_array)
t1 = time.perf_counter()
identify_exposed_voxels(random_array)
t2 = time.perf_counter()
identify_exposed_voxels(random_array)
t3 = time.perf_counter()

print("identify_exposed_voxels0", t1-t0)
print("identify_exposed_voxels", t2-t1)
print("identify_exposed_voxels", t3-t2)