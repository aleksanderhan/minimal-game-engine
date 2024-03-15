import numpy as np

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




import heapq
from direct.stdpy.threading import Lock

class TaskWrapper:

    def __init__(self, task: tuple[int, int], priority: int):
        self.task = task
        self.priority = priority

class ReprioritizationQueue:

    def __init__(self):
        self.queue: list[tuple[int, int, int]] = []
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

    def batch_change_priority0(self, wrapped_tasks: list[TaskWrapper]):
        with self.lock:
            # Temporarily store modified wrappers to re-add them after modifications
            temp_wrappers = []
            
            # Find and adjust the priority of the task
            for _, counter, wrapper in self.queue:
                for reprioritized_wrapper in wrapped_tasks:
                    if wrapper.task == reprioritized_wrapper.task:
                        # Update the wrapper priority and store it for re-adding
                        temp_wrappers.append((reprioritized_wrapper.priority, reprioritized_wrapper))
            
            # If we found and adjusted any wrappers, rebuild the heap
            if temp_wrappers:
                # Remove the old entry and add the updated one
                self.queue = [(priority, counter, wrapper) for priority, counter, wrapper in self.queue if wrapper.task != reprioritized_wrapper.task]
                for priority, wrapper in temp_wrappers:
                    heapq.heappush(self.queue, (priority, counter, wrapper))

    def batch_change_priority1(self, wrapped_tasks: list[TaskWrapper]):
        with self.lock:
            # Create a mapping from task to new priority for quick lookup
            new_priorities = {wrapper.task: wrapper.priority for wrapper in wrapped_tasks}

            # Iterate over the queue and update priorities based on new_priorities
            for i, (_, counter, wrapper) in enumerate(self.queue):
                if wrapper.task in new_priorities:
                    # Update priority directly in the queue
                    self.queue[i] = (new_priorities[wrapper.task], counter, wrapper)

            # After updating the priorities in the queue, re-heapify to maintain the heap invariant
            heapq.heapify(self.queue)

    def batch_change_priority2(self, wrapped_tasks: list[TaskWrapper]):
        with self.lock:
            # Create a dictionary to map tasks to new priorities
            priority_map = {wt.task: wt.priority for wt in wrapped_tasks}

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







import time

t0 = time.perf_counter()
queue = ReprioritizationQueue()

queue.put((1, 1), 1)
queue.put((2, 2), 2)
queue.put((3, 3), 3)
queue.put((4, 4), 4)
queue.put((5, 5), 5)

change_priority = [TaskWrapper((1, 1), 7), TaskWrapper((2, 2), 8)]
queue.batch_change_priority0(change_priority)
task0 = queue.get()
print("task0", task0)
task1 = queue.get()
print("task1", task1)
task2 = queue.get()
print("task2", task2)
dt = time.perf_counter() - t0
print(dt)
print()

t0 = time.perf_counter()
queue = ReprioritizationQueue()

queue.put((1, 1), 1)
queue.put((2, 2), 2)
queue.put((3, 3), 3)
queue.put((4, 4), 4)
queue.put((5, 5), 5)

change_priority = [TaskWrapper((1, 1), 7), TaskWrapper((2, 2), 8)]
queue.batch_change_priority1(change_priority)
task0 = queue.get()
print("task0", task0)
task1 = queue.get()
print("task1", task1)
task2 = queue.get()
print("task2", task2)
dt = time.perf_counter() - t0
print(dt)
print()

t0 = time.perf_counter()
queue = ReprioritizationQueue()

queue.put((1, 1), 1)
queue.put((2, 2), 2)
queue.put((3, 3), 3)
queue.put((4, 4), 4)
queue.put((5, 5), 5)

change_priority = [TaskWrapper((1, 1), 7), TaskWrapper((2, 2), 8)]
queue.batch_change_priority2(change_priority)
task0 = queue.get()
print("task0", task0)
task1 = queue.get()
print("task1", task1)
task2 = queue.get()
print("task2", task2)
dt = time.perf_counter() - t0
print(dt)
print()
