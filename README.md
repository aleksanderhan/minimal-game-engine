# minimal-game-engine

Minimal voxel engine with breakable physics, written in python.

## Instructions

### Prerequisites

* git - to checkout this repo
* python 3.10 and pip - to run the program and install dependencies

### Install

* clone this repo
* install dependencies: `pip install -r requirements.txt`

### How to run

* Run it like so in the terminal: `python game_engine.py` for a procedurally generated world experience.
* possibly try to run like this:
    `__NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __VK_LAYER_NV_optimus=NVIDIA_only \
    python game_engine.py`



### Controls

Use the keys `wasd` to steer the position of the camera and use the mouse to steer the angle of the camera.

## Demo
### Collision physics:
![Semi-autogpt example](docs/demo.gif)

### Gravity:
![Semi-autogpt example](docs/demo2.gif)

### Multi-collision physics:
![Semi-autogpt example](docs/demo3.gif)

### Procedural generated terrain:
![Semi-autogpt example](docs/demo4.gif)

### Place blocks:
![Semi-autogpt example](docs/demo5.gif)