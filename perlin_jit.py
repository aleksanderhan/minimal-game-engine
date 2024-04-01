import numpy as np
import numba as nb
import random

@nb.njit(nopython=True, cache=True, nogil=True, fastmath=True)
def _fade(t):
    """Fade function as defined by Ken Perlin. This eases coordinate values
    so that they will ease towards integral values. This helps to avoid
    artifacts and sharp transitions."""
    return 6*t**5 - 15*t**4 + 10*t**3

@nb.njit(nopython=True, cache=True, nogil=True, fastmath=True)
def _lerp(a, b, x):
    """Linear interpolation between a and b with x as the fraction."""
    return a + x * (b - a)

@nb.njit(nopython=True, cache=True, nogil=True, fastmath=True)
def _gradient(h, x, y):
    """Calculate gradient vector dot product with (x,y) vector."""
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    # Directly index into g without using :, since it's effectively a 1D array here
    return g[0] * x + g[1] * y

@nb.njit(nopython=True, cache=True, nogil=True, parallel=True)
def pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=10000, repeaty=10000, base=0):
    max_x, max_y = repeatx, repeaty
    total = 0
    frequency = 1
    amplitude = 1
    maxAmplitude = 0
    for _ in range(octaves):
        n = _perlin(x * frequency, y * frequency, max_x, max_y, base)
        total += n * amplitude
        maxAmplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    return total / maxAmplitude

@nb.njit(nopython=True, cache=True, nogil=True, parallel=True)
def _perlin(x, y, max_x, max_y, base):
    """A single octave of perlin noise."""
    # Determine grid cell coordinates
    xi = np.int64(np.floor(x))
    yi = np.int64(np.floor(y))
        
    # Relative x, y position in grid cell
    xf = x - xi
    yf = y - yi
    
    # Wrap around
    xi %= max_x
    yi %= max_y
    
    # Fade curves
    u = _fade(xf)
    v = _fade(yf)
    
    # Hash coordinates for gradient directions
    random.seed(base + xi + max_x * yi)
    n00 = random.randint(0, 4)
    random.seed(base + xi + max_x * (yi + 1))
    n01 = random.randint(0, 4)
    random.seed(base + (xi + 1) + max_x * yi)
    n10 = random.randint(0, 4)
    random.seed(base + (xi + 1) + max_x * (yi + 1))
    n11 = random.randint(0, 4)
    
    # The gradients
    g00 = _gradient(n00, xf, yf)
    g01 = _gradient(n01, xf, yf-1)
    g10 = _gradient(n10, xf-1, yf)
    g11 = _gradient(n11, xf-1, yf-1)
    
    # Interpolate
    return _lerp(
            _lerp(g00, g10, u),
            _lerp(g01, g11, u),
            v)
