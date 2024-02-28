# minimal-game-engine

Minimal game engine written in python.

## Instructions

### Prerequisites

* git - to checkout this repo
* python 3.10 and pip - to run the program and install dependencies

### Install

* clone this repo
* install dependencies: `pip install -r requirements.txt`

### How to run

* Run it like so in the terminal: `python game_engine.py` for a flat world experience.
* Run it with the terrain flag: `python game_engine.py --terrain=perlin` for a world generated with perlin noise, else `flat`
* Run it with the texture flag: `python game_engine.py --texture=grass` for a grass textured world, else `chess`

### Controls

Use the keys `a`,`s`,`d`,`w` to steer the position of the camera and use the mouse to steer the angle of the camera. Use left and right mouse click to shot
small and big bullets respectively. Use `f` key to place blocks.

## Demo
### Collision physics:
![Semi-autogpt example](docs/demo.gif)

### Gravity:
![Semi-autogpt example](docs/demo2.gif)

### Many-body physics:
![Semi-autogpt example](docs/demo3.gif)

### Procedural generated terrain:
![Semi-autogpt example](docs/demo4.gif)

### Place blocks:
![Semi-autogpt example](docs/demo5.gif)