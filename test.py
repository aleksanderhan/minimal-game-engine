import numpy as np
import numba as nb
import time
import heapq
from direct.stdpy.threading import RLock
from typing import Any



# Pad the array with 0's on all sides
voxel_array = np.ones((1, 1, 1), np.int8)
voxel_array = np.pad(voxel_array, pad_width=[(1, 1)] * 3, mode='constant', constant_values=0)

print(voxel_array)