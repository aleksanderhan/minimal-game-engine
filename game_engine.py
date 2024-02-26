import numpy as np
import noise
import pyautogui
import argparse

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
from direct.actor.Actor import Actor

from panda3d.core import (
    AmbientLight, CardMaker, DirectionalLight, Geom, GeomNode, GeomTriangles,
    GeomVertexData, GeomVertexFormat, GeomVertexWriter, KeyboardButton, LColor,
    LineSegs, Material, NodePath, TextNode, Vec3, Vec4, WindowProperties,
    loadPrcFileData
)
from panda3d.bullet import (
    BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletHeightfieldShape  # Conceptual
)
from panda3d.bullet import BulletRigidBodyNode, BulletCapsuleShape
from panda3d.core import NodePath
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletSphereShape, BulletCylinderShape, BulletHingeConstraint, BulletDebugNode
from panda3d.core import Vec3, TransformState
from math import cos, sin, radians
from panda3d.core import Texture
import random
from helper import build_robot
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletGenericConstraint
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import TransparencyAttrib

#random.seed()

loadPrcFileData("", "load-file-type p3assimp")


class ChunkManager:
    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.loaded_chunks = {}

    def get_player_chunk_pos(self):
        player_pos = self.game_engine.camera.getPos()
        chunk_x = int(player_pos.x) // self.game_engine.chunk_size
        chunk_y = int(player_pos.y) // self.game_engine.chunk_size
        return chunk_x, chunk_y

    def update_chunks(self):
        chunk_x, chunk_y = self.get_player_chunk_pos()
        # Adjust the range to load chunks further out by one additional level
        for x in range(chunk_x - 3, chunk_x + 3):  # Increase the range by one on each side
            for y in range(chunk_y - 3, chunk_y + 3):  # Increase the range by one on each side
                if (x, y) not in self.loaded_chunks:
                    self.load_chunk(x, y)
        # Adjust the identification of chunks to unload, if necessary
        for chunk_pos in list(self.loaded_chunks.keys()):
            if abs(chunk_pos[0] - chunk_x) > 3 or abs(chunk_pos[1] - chunk_y) > 3:  # Adjusted range
                self.unload_chunk(*chunk_pos)

    def load_chunk(self, chunk_x, chunk_y):
        # Generate the chunk and obtain both visual (terrainNP) and physics components (terrainNode)
        terrainNP, terrainNode = self.game_engine.generate_chunk(chunk_x, chunk_y)
        
        # Store both components in the loaded_chunks dictionary
        self.loaded_chunks[(chunk_x, chunk_y)] = (terrainNP, terrainNode)

    def unload_chunk(self, chunk_x, chunk_y):
        chunk_data = self.loaded_chunks.pop((chunk_x, chunk_y), None)
        if chunk_data:
            terrainNP, terrainNode = chunk_data
            # Remove the visual component from the scene
            terrainNP.removeNode()
            # Remove the physics component from the physics world
            self.game_engine.physicsWorld.removeRigidBody(terrainNode)


class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.ground_level = 0 
        self.chunk_size = 64
        
        self.chunk_manager = ChunkManager(self)
        self.voxel_positions = set()

        self.camera.setPos(0, -30, 30)
        self.camera.lookAt(0, 0, 0)

        self.setupBulletPhysics()
        self.setup_environment()
        self.setup_lighting()
        self.setup_crosshair()

        self.firstUpdate = True  # Add this line to initialize firstUpdate

        self.setup_movement_controls()
        self.init_fps_counter()
        self.init_mouse_control()

        self.pitch = 0  # Initialize a variable to keep track of the camera's pitch
        self.heading = 0  # Initialize a variable to keep track of the camera's heading

        self.taskMgr.add(self.move_camera_task, "MoveCameraTask")
        self.taskMgr.add(self.update_fps_counter, "UpdateFPSTask")
        self.taskMgr.add(self.mouse_task, "MouseTask")
        self.taskMgr.add(self.updatePhysics, "updatePhysics")
        self.taskMgr.add(self.update, "update")

        self.accept('mouse1', self.shoot_bullet)  # Listen for left mouse click
        self.accept('mouse3', self.shoot_big_bullet)
        self.accept('f', self.create_and_place_voxel)

    def setup_crosshair(self):
        # Path to the crosshair image
        crosshair_image = 'assets/crosshair.png'
        
        # Create and position the crosshair at the center of the screen
        self.crosshair = OnscreenImage(image=crosshair_image, pos=(0, 0, 0))
        self.crosshair.setTransparency(TransparencyAttrib.MAlpha)
        self.crosshair.setScale(0.05, 1, 0.05)

    def create_and_place_voxel(self):
        # Adjust the multiplier to set the voxel 1 game meter away from the camera
        position = self.camera.getPos() + self.camera.getQuat().getForward() * 1  # Set distance to 1 game meter
        # Round the position to snap to grid
        grid_position = Vec3(round(position.x), round(position.y), round(position.z))

        # Check if a voxel already exists at this position
        if not self.voxel_exists_at(grid_position):
            self.create_voxel(grid_position, scale=0.2)
            self.voxel_positions.add((grid_position.x, grid_position.y, grid_position.z))
        else:
            # Optional: Handle the case where a voxel already exists at the intended position
            pass

    def voxel_exists_at(self, position):
        # Convert Vec3 position to a tuple for set comparison
        return (position.x, position.y, position.z) in self.voxel_positions

    def find_nearest_voxel_position(self, position):
        # This is a simplified version. You might need to check for actual nearest positions
        # based on your game's logic, e.g., checking adjacent positions for existing voxels.
        adjustments = [Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1)]
        for adj in adjustments:
            adj_position = position + adj
            if self.voxel_exists_at(adj_position):
                return adj_position
        return None



    def create_voxel(self, position, scale=0.2):
        voxel_shape = BulletBoxShape(Vec3(scale / 2, scale / 2, scale / 2))
        voxel_node = BulletRigidBodyNode('Voxel')
        voxel_node.addShape(voxel_shape)
        # Check if the voxel is placed on the ground
        if position.z == self.ground_level:  # Assuming you have a defined ground_level variable
            voxel_node.setMass(0)  # Make the voxel static if it's on the ground
        else:
            voxel_node.setMass(1.0)  # Otherwise, it's dynamic
        voxel_np = self.render.attachNewNode(voxel_node)
        voxel_np.setPos(position)
        self.physicsWorld.attachRigidBody(voxel_node)
        voxel_model = self.loader.loadModel("models/box.egg")
        voxel_model.setScale(scale)
        voxel_model.reparentTo(voxel_np)
        voxel_model.setColor(0.5, 0.5, 0.5, 1)
        return voxel_node

    
    def update(self, task):
        self.chunk_manager.update_chunks()
        return Task.cont


    def generate_chunk(self, chunk_x, chunk_y):
        # Generate the height map for this chunk
        if self.args.terrain == "perlin":
            height_map = self.generate_perlin_height_map(chunk_x, chunk_y)
        else:
            height_map = self.generate_flat_height_map(self.chunk_size)

        # Generate mesh data from the height map
        vertices, indices = self.create_mesh_data(height_map, self.chunk_size, 5)  # Assume height scale is 5

        # Apply texture based on args
        texture_path = "assets/grass.png" if self.args.texture == "grass" else "assets/chess.png"
        # Create terrain geometry and apply texture
        terrainNP = self.apply_texture_to_terrain(self.chunk_size, vertices, indices, texture_path)
        terrainNP.reparentTo(self.render)

        # Position the terrain chunk according to its world coordinates
        world_x = chunk_x * self.chunk_size
        world_y = chunk_y * self.chunk_size
        terrainNP.setPos(world_x, world_y, 0)

        # Add physics
        terrainNode = self.add_mesh_to_physics(vertices, indices, world_x, world_y)

        return terrainNP, terrainNode


    def setup_lighting(self):
        self.setBackgroundColor(0.53, 0.81, 0.98, 1)  # Set the background to light blue
        # Ambient Light
        ambient_light = AmbientLight('ambient_light')
        ambient_light.setColor((0.2, 0.2, 0.2, 1))
        ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_light_np)
        
        # Directional Light
        directional_light = DirectionalLight('directional_light')
        directional_light.setColor((0.8, 0.8, 0.8, 1))
        directional_light_np = self.render.attachNewNode(directional_light)
        directional_light_np.setHpr(0, -60, 0)
        self.render.setLight(directional_light_np)


    def setupBulletPhysics(self):
        self.physicsWorld = BulletWorld()
        self.physicsWorld.setGravity(Vec3(0, 0, -9.81))

        if self.args.debug:
            debugNode = BulletDebugNode('Debug')
            debugNP = render.attachNewNode(debugNode)
            debugNP.show()
            self.physicsWorld.setDebugNode(debugNP.node())


    def updatePhysics(self, task):
        dt = globalClock.getDt()
        self.physicsWorld.doPhysics(dt)
        
        return Task.cont
    
    def shoot_bullet(self):
        if self.mouseWatcherNode.hasMouse():
            # Get the mouse position in the world
            mpos = self.mouseWatcherNode.getMouse()
            
            # Use the camera's position and orientation to shoot the bullet
            position = self.camera.getPos()
            direction = self.camera.getQuat().getForward()  # Get the forward direction of the camera
            velocity = direction * 100  # Adjust the speed as necessary
            
            # Create and shoot the bullet
            self.create_bullet(position, velocity)
    
    def shoot_big_bullet(self):
        if self.mouseWatcherNode.hasMouse():
            # Get the mouse position in the world
            mpos = self.mouseWatcherNode.getMouse()
            
            # Use the camera's position and orientation to shoot the bullet
            position = self.camera.getPos()
            direction = self.camera.getQuat().getForward()  # Get the forward direction of the camera
            velocity = direction * 30  # Adjust the speed as necessary
            
            # Create and shoot the bullet
            self.create_bullet(position, velocity, True)


    def create_bullet(self, position, velocity, big_bullet=False):
        # Bullet model
        bullet_model = self.loader.loadModel("models/misc/sphere.egg")  # Use a simple sphere model
        bullet_node = BulletRigidBodyNode('Bullet')
        
        # Bullet physics
        if big_bullet:
            bullet_model.setScale(1)  # Scale down to bullet size
            bullet_shape = BulletSphereShape(1)  # The collision shape radius
            bullet_node.setMass(10) 
            bullet_model.setColor(1, 0, 0, 1)
        else:
            bullet_model.setScale(0.2)  # Scale down to bullet size
            bullet_shape = BulletSphereShape(0.2)  # The collision shape radius
            bullet_node.setMass(0.1)
        
        
        bullet_node.addShape(bullet_shape)
        bullet_node.setLinearVelocity(velocity)  # Set initial velocity
        
        bullet_np = self.render.attachNewNode(bullet_node)
        bullet_np.setPos(position)
        bullet_model.reparentTo(bullet_np)
        
        self.physicsWorld.attachRigidBody(bullet_node)
        
        return bullet_np

    
    def setup_environment(self):
        # Create the terrain mesh (both visual and physical)
        self.create_sphere((32, 32, 10))
        self.create_sphere((32, 32, 15))
        self.create_sphere((32, 32, 20))
        self.create_sphere((32, 32, 25))
        #build_robot(self.physicsWorld)


    def apply_texture_to_terrain(self, board_size, vertices, indices, texture_path):
        # Load the texture
        terrainTexture = loader.loadTexture(texture_path)
        if terrainTexture:
            print("Texture loaded successfully.")
        else:
            print("Failed to load texture.")

        terrainTexture.setWrapU(Texture.WMRepeat)
        terrainTexture.setWrapV(Texture.WMRepeat)

        # Create the terrain geometry
        format = GeomVertexFormat.getV3n3t2()  # Format including normals and texture coordinates
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)

        # Writers for data
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        texcoord = GeomVertexWriter(vdata, 'texcoord')

        # Assuming you have a method to calculate UVs correctly based on vertices
        uv_coords = self.calculate_uv_coordinates(vertices, board_size)

        # Add vertices, normals, and UVs to the vertex data
        for i, v in enumerate(vertices):
            vertex.addData3f(v[0], v[1], v[2])
            normal.addData3f(0, 0, 1)  # Simplified, should be calculated based on terrain
            texcoord.addData2f(uv_coords[i][0], uv_coords[i][1])

        # Create triangles
        prim = GeomTriangles(Geom.UHStatic)
        for i in range(0, len(indices), 3):
            prim.addVertices(indices[i], indices[i+1], indices[i+2])
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('terrain')
        node.addGeom(geom)
        terrainNP = NodePath(node)
        terrainNP.reparentTo(self.render)

        # Apply the texture
        terrainNP.setTexture(terrainTexture)
        return terrainNP

    def calculate_uv_coordinates(self, vertices, board_size):
        # Simple UV mapping: map vertex position to UV coordinates
        uv_coords = []
        for v in vertices:
            u = v[0] / board_size
            v = v[1] / board_size
            uv_coords.append((u, v))
        return uv_coords



    def create_mesh_data(self, height_map, board_size, height_scale):
        # Adjust the size for seamless edges
        adjusted_size = board_size + 1  # Adjust for the extra row/column

        # Generate meshgrid with the adjusted size
        x, y = np.meshgrid(np.arange(adjusted_size), np.arange(adjusted_size), indexing='ij')

        # Ensure z has the correct shape, assuming height_map is already (board_size + 1, board_size + 1)
        z = height_map * height_scale

        # Now x, y, and z have matching shapes, and you can safely stack them
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

        # Adjust index calculation for the extra vertices
        indices = []
        for iy in range(board_size):
            for ix in range(board_size):
                # Calculate indices for two triangles covering the quad
                indices += [
                    iy * adjusted_size + ix, (iy + 1) * adjusted_size + ix, iy * adjusted_size + (ix + 1),
                    iy * adjusted_size + (ix + 1), (iy + 1) * adjusted_size + ix, (iy + 1) * adjusted_size + (ix + 1)
                ]

        return vertices, np.array(indices, dtype=np.int32)

    def add_mesh_to_physics(self, vertices, indices, world_x, world_y):
        terrainMesh = BulletTriangleMesh()
        for i in range(0, len(indices), 3):
            v0, v1, v2 = vertices[indices[i]], vertices[indices[i+1]], vertices[indices[i+2]]
            terrainMesh.addTriangle(Vec3(*v0), Vec3(*v1), Vec3(*v2))

        terrainShape = BulletTriangleMeshShape(terrainMesh, dynamic=False)
        terrainNode = BulletRigidBodyNode('Terrain')
        terrainNode.addShape(terrainShape)
        terrainNP = self.render.attachNewNode(terrainNode)
        # Set the position of the terrain's physics node to match its visual representation
        terrainNP.setPos(world_x, world_y, 0)
        self.physicsWorld.attachRigidBody(terrainNode)
        return terrainNode

    def generate_perlin_height_map(self, chunk_x, chunk_y):
        scale = 0.1  # Adjust scale to control the "zoom" level of the noise
        octaves = 4  # Number of layers of noise to combine
        persistence = 0.5  # Amplitude of each octave
        lacunarity = 2.0  # Frequency of each octave

        height_map = np.zeros((self.chunk_size + 1, self.chunk_size + 1))

        # Calculate global offsets
        global_offset_x = chunk_x * self.chunk_size
        global_offset_y = chunk_y * self.chunk_size

        for x in range(self.chunk_size + 1):
            for y in range(self.chunk_size + 1):
                # Calculate global coordinates
                global_x = (global_offset_x + x) * scale
                global_y = (global_offset_y + y) * scale

                # Generate height using Perlin noise
                height = noise.pnoise2(global_x, global_y,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=10000,  # Large repeat region to avoid repetition
                                    repeaty=10000,
                                    base=0)  # Base can be any constant, adjust for different terrains

                # Map the noise value to a desired height range if needed
                height_map[x, y] = height

        return height_map
    
    def generate_flat_height_map(self, board_size, height=0):
        # Adjust board_size to account for the extra row and column for seamless edges
        adjusted_size = board_size + 1
        # Create a 2D NumPy array filled with the specified height value
        height_map = np.full((adjusted_size, adjusted_size), height)
        return height_map
    
    def create_sphere(self, position):
        # Sphere physics
        sphereShape = BulletSphereShape(1)  # Match this with the scale of your sphere model
        sphereNode = BulletRigidBodyNode('Sphere')
        sphereNode.addShape(sphereShape)
        sphereNode.setMass(10.0)
        sphereNP = self.render.attachNewNode(sphereNode)
        sphereNP.setPos(*position)  # Adjust the height to see it fall
        self.physicsWorld.attachRigidBody(sphereNode)

        # Load the sphere model and attach it to the physics node
        sphere = self.loader.loadModel("models/misc/sphere.egg")
        sphere.reparentTo(sphereNP)  # Correctly attach the model to the NodePath
        sphere.setScale(1)  # Adjust the scale as needed
        sphere.setColor(1, 0, 0, 1)  # Set the sphere's color

    def init_mouse_control(self):
        """Initial setup for mouse control."""
        self.disableMouse()  # Disable the default mouse camera control
        self.mouseSpeedX = 100  # Adjust as needed
        self.mouseSpeedY = 100  # Adjust as needed
        self.lastMouseX = 0
        self.lastMouseY = 0
        self.cameraPitch = 0
        self.cameraHeading = 0
    
    def mouse_task(self, task):
        if self.mouseWatcherNode.hasMouse():
            mouseX, mouseY = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
            
            # Calculate deltas as before
            deltaX = mouseX - self.lastMouseX
            deltaY = mouseY - self.lastMouseY
            
            # Update camera orientation
            self.cameraHeading -= deltaX * self.mouseSpeedX
            self.cameraPitch = max(min(self.cameraPitch + deltaY * self.mouseSpeedY, 90), -90)
            
            self.camera.setHpr(self.cameraHeading, self.cameraPitch, 0)
            
            # Check if cursor is near the edge, then re-center it
            screenWidth, screenHeight = pyautogui.size()
            currentMouseX, currentMouseY = pyautogui.position()
            
            if currentMouseX <= 1 or currentMouseX >= screenWidth - 2 or currentMouseY <= 1 or currentMouseY >= screenHeight - 2:
                # Move cursor to the center of the screen
                pyautogui.moveTo(screenWidth / 2, screenHeight / 2)
                self.lastMouseX, self.lastMouseY = 0, 0  # Reset last mouse position to the center
            else:
                self.lastMouseX, self.lastMouseY = mouseX, mouseY

        return Task.cont


    def init_fps_counter(self):
        """Initializes the FPS counter on the screen."""
        self.fps_counter = OnscreenText(text="FPS: 0", pos=(-1.3, 0.9), scale=0.07,
                                        fg=(1, 1, 1, 1), align=TextNode.ALeft)

    def update_fps_counter(self, task):
        """Updates the FPS counter with the current frame rate."""
        fps = round(globalClock.getAverageFrameRate(), 1)
        self.fps_counter.setText(f"FPS: {fps}")
        return Task.cont

    def setup_movement_controls(self):
        # WASD keys for forward, left, backward, right movement
        inputState.watchWithModifiers('forward', KeyboardButton.asciiKey('w'))
        inputState.watchWithModifiers('left', KeyboardButton.asciiKey('a'))
        inputState.watchWithModifiers('backward', KeyboardButton.asciiKey('s'))
        inputState.watchWithModifiers('right', KeyboardButton.asciiKey('d'))
        
        # Space for up, Alt for down
        inputState.watchWithModifiers('up', KeyboardButton.space())
        inputState.watchWithModifiers('down', KeyboardButton.alt())

        # Additional keys for rotating the camera horizontally
        inputState.watchWithModifiers('rotateLeft', KeyboardButton.asciiKey('q'))
        inputState.watchWithModifiers('rotateRight', KeyboardButton.asciiKey('e'))

    def move_camera_task(self, task):
        dt = globalClock.getDt()
        speed = 20  # Existing movement speed
        lift_speed = 10  # Existing up and down speed
        rotate_speed = 50  # Speed for rotating the camera, adjust as needed

        # Lateral movement
        if inputState.isSet('forward'):
            self.camera.setY(self.camera, speed * dt)
        if inputState.isSet('backward'):
            self.camera.setY(self.camera, -speed * dt)
        if inputState.isSet('left'):
            self.camera.setX(self.camera, -speed * dt)
        if inputState.isSet('right'):
            self.camera.setX(self.camera, speed * dt)

        # Vertical movement
        if inputState.isSet('up'):
            self.camera.setZ(self.camera, lift_speed * dt)
        if inputState.isSet('down'):
            self.camera.setZ(self.camera, -lift_speed * dt)

        # Horizontal rotation
        if inputState.isSet('rotateLeft'):
            self.camera.setH(self.camera.getH() + rotate_speed * dt)
        if inputState.isSet('rotateRight'):
            self.camera.setH(self.camera.getH() - rotate_speed * dt)

        return Task.cont


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--terrain', action='store', default="flat")
    parser.add_argument('--texture', action='store', default="chess")
    parser.add_argument('--debug', action="store_true", default=False)
    args = parser.parse_args()

    game = GameEngine(args)
    game.run()