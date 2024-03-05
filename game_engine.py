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
    loadPrcFileData, GeomVertexReader
)
from panda3d.bullet import (
    BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletHeightfieldShape
)
from panda3d.bullet import BulletRigidBodyNode, BulletCapsuleShape
from panda3d.core import NodePath, Point3
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletSphereShape, BulletCylinderShape, BulletHingeConstraint, BulletDebugNode
from panda3d.core import Vec3, TransformState
from math import cos, sin, radians
from panda3d.core import Texture
import random
from helper import build_robot, toggle
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletGenericConstraint
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import TransparencyAttrib
from functools import lru_cache 
from panda3d.core import WindowProperties
from panda3d.core import CollisionNode, CollisionSphere, CollisionBox, CollisionTraverser, CollisionHandlerEvent
from panda3d.core import BitMask32
from panda3d.bullet import BulletGhostNode
from scipy.interpolate import interp2d
from PIL import Image

random.seed()

loadPrcFileData("", "load-file-type p3assimp")
loadPrcFileData("", "bullet-enable-contact-events true")

class ChunkManager:
    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.loaded_chunks = {}

    def get_player_chunk_pos(self):
        player_pos = self.game_engine.camera.getPos()
        chunk_x = int(player_pos.x) // self.game_engine.chunk_size
        chunk_y = int(player_pos.y) // self.game_engine.chunk_size
        return chunk_x, chunk_y

    def update_chunks(self, levels=2):
        chunk_x, chunk_y = self.get_player_chunk_pos()
        # Adjust the range to load chunks further out by one additional level
        for x in range(chunk_x - levels, chunk_x + levels):  # Increase the range by one on each side
            for y in range(chunk_y - levels, chunk_y + levels):  # Increase the range by one on each side
                if (x, y) not in self.loaded_chunks:
                    self.load_chunk(x, y)
        # Adjust the identification of chunks to unload, if necessary
        for chunk_pos in list(self.loaded_chunks.keys()):
            if abs(chunk_pos[0] - chunk_x) > levels or abs(chunk_pos[1] - chunk_y) > levels:  # Adjusted range
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
        self.scale = 0.5
        self.ground_height = 0

        self.chunk_size = 4
        self.chunk_manager = ChunkManager(self)
        self.voxel_world_map = {}
        self.texture_paths = {
            "stone": "assets/stone.jpeg",
            "grass": "assets/grass.png"
        }

        self.camera.setPos(0, 0, 10)
        self.camera.lookAt(5, 5, 0)

        self.setup_physics()
        self.setup_environment()
        self.setup_lighting()
        self.setup_crosshair()
        self.setup_movement_controls()
        self.init_fps_counter()
        self.init_mouse_control()

        self.taskMgr.add(self.move_camera_task, "MoveCameraTask")
        self.taskMgr.add(self.update_fps_counter, "UpdateFPSTask")
        self.taskMgr.add(self.mouse_task, "MouseTask")
        self.taskMgr.add(self.update_physics, "UpdatePhysics")
        self.taskMgr.add(self.update_terrain, "UpdateTerrain")

        self.accept('mouse1', self.shoot_bullet)  # Listen for left mouse click
        self.accept('mouse3', self.shoot_big_bullet)
        self.accept('r', self.manual_raycast_test)
        self.accept('g', self.toggle_gravity)

    def setup_environment(self):
        #build_robot(self.physicsWorld)
        pass
    
    def manual_raycast_test(self):
        result = self.cast_ray_from_camera(10000)
        if result.hasHit():
            print("Hit at:", result.getHitPos())
        else:
            print("No hit detected.")

    def cast_ray_from_camera(self, distance=10):
        """Casts a ray from the camera to detect voxels."""
        # Get the camera's position and direction
        cam_pos = self.camera.getPos()
        cam_dir = self.camera.getQuat().getForward()
        
        # Define the ray's start and end points (within a certain distance)
        start_point = cam_pos
        end_point = cam_pos + cam_dir * distance  # Adjust the distance as needed
        
        # Perform the raycast
        return self.physicsWorld.rayTestClosest(start_point, end_point)
        
    def get_voxel_world(self, chunk_x, chunk_y):
        # Dimensions of the voxel world
        max_height = 10
        width = self.chunk_size
        depth = self.chunk_size

        if self.voxel_world_map.get((chunk_x, chunk_y)) is None:
            heightmap = self.generate_perlin_height_map(chunk_x, chunk_y)

            # Initialize a 3D numpy array with zeros (air)
            voxel_world = np.zeros((width, depth, max_height), dtype=np.uint8)

            voxel_world[:, :, 0] = 1 # setting bedrock to be all ground

            # Fill the voxel world based on the heightmap
            for x in range(width):
                for y in range(depth):
                    ground_height = int(heightmap[x, y] * max_height)
                    for z in range(1, ground_height-1):
                        # Randomly choose between two types of ground (1 or 2)
                        voxel_world[x, y, z] = np.random.choice([1, 2], p=[0.5, 0.5])

            self.voxel_world_map[(chunk_x, chunk_y)] = voxel_world
            return voxel_world

        return self.voxel_world_map.get((chunk_x, chunk_y))
        
    def generate_chunk(self, chunk_x, chunk_y):
        voxel_world = self.get_voxel_world(chunk_x, chunk_y)
        
        vertices, indices = self.create_mesh_data(voxel_world)
        terrainNP = self.apply_textures_to_voxels(voxel_world, vertices, indices)

        # Position the flat terrain chunk according to its world coordinates
        world_x = chunk_x * self.chunk_size * self.scale
        world_y = chunk_y * self.chunk_size * self.scale
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

    def setup_physics(self):
        self.acceleration_due_to_gravity = toggle(Vec3(0, 0, self.args.g), Vec3(0, 0, 0))
        self.physicsWorld = BulletWorld()
        self.physicsWorld.setGravity(next(self.acceleration_due_to_gravity))

        if self.args.debug:
            debugNode = BulletDebugNode('Debug')
            debugNP = self.render.attachNewNode(debugNode)
            debugNP.show()
            self.physicsWorld.setDebugNode(debugNP.node())

    def toggle_gravity(self):
        self.physicsWorld.setGravity(next(self.acceleration_due_to_gravity))
    
    def shoot_bullet(self, speed=100, scale=0.2, mass=0.1, color=(1, 1, 1, 1)):
        # Use the camera's position and orientation to shoot the bullet
        position = self.camera.getPos()
        direction = self.camera.getQuat().getForward()  # Get the forward direction of the camera
        velocity = direction * speed  # Adjust the speed as necessary
        
        # Create and shoot the bullet
        self.create_bullet(position, velocity, scale, mass, color)

    def shoot_big_bullet(self):
        return self.shoot_bullet(30, 0.5, 10, (1, 0, 0, 1))

    def create_bullet(self, position, velocity, scale, mass, color):
        # Bullet model
        bullet_model = self.loader.loadModel("models/misc/sphere.egg")  # Use a simple sphere model
        bullet_node = BulletRigidBodyNode('Bullet')
        
        # Bullet physics
        bullet_model.setScale(scale)  # Scale down to bullet size
        bullet_shape = BulletSphereShape(scale)  # The collision shape radius
        bullet_node.setMass(mass) 
        bullet_model.setColor(*color)
        
        bullet_node.addShape(bullet_shape)
        bullet_node.setLinearVelocity(velocity)  # Set initial velocity
        
        bullet_np = self.render.attachNewNode(bullet_node)
        bullet_np.setPos(position)
        #bullet_np.node().setCcdMotionThreshold(1e-7)
        #bullet_np.node().setCcdSweptSphereRadius(0.50)
        bullet_model.reparentTo(bullet_np)
        
        self.physicsWorld.attachRigidBody(bullet_node)
        
        return bullet_np

    def apply_textures_to_voxels(self, voxel_world, vertices, indices, voxel_size=0.5):
        # Assuming `vertices` is a flat list of vertex positions (x, y, z)
        # and `indices` defines which vertices form each triangle.

        # Load your texture atlas
        texture_atlas = self.loader.loadTexture("texture_atlas.png")

        # Create a new GeomVertexFormat that includes texture coordinates
        format = GeomVertexFormat.getV3n3cpt2()  # Position, Normal, Color, Texture coordinates
        vdata = GeomVertexData('voxel_data', format, Geom.UHStatic)
        
        # Create writers for vertices, normals (optional), colors (optional), and texture coordinates
        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        normal_writer = GeomVertexWriter(vdata, 'normal')
        texcoord_writer = GeomVertexWriter(vdata, 'texcoord')

        # Example: Add vertices and UVs for each face
        for i in range(0, len(indices), 6):  # 6 indices per quad (two triangles)
            for j in range(6):
                idx = indices[i + j]
                vertex_writer.addData3f(vertices[idx * 3], vertices[idx * 3 + 1], vertices[idx * 3 + 2])
                # Add normal data if you have it
                # normal_writer.addData3f(...)
                # Compute and add texture coordinates based on the voxel type and face
                # This is simplified; you'll need to adjust based on your atlas layout
                u, v = 0, 0  # Placeholder: Compute these based on the face and voxel type
                texcoord_writer.addData2f(u, v)

        # Create the GeomTriangles object
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(0, len(indices), 3):
            tris.addVertices(indices[i], indices[i+1], indices[i+2])

        # Combine everything into a Geom
        geom = Geom(vdata)
        geom.addPrimitive(tris)

        # Attach the Geom to a GeomNode, and the GeomNode to a NodePath
        geom_node = GeomNode('voxel_geom')
        geom_node.addGeom(geom)
        geom_np = NodePath(geom_node)
        geom_np.setTexture(texture_atlas)

        # Attach the NodePath to your scene graph where appropriate
        geom_np.reparentTo(self.render)

        # Return the NodePath, in case it's needed
        return geom_np

    def visualize_normals(self, geom_node, scale=0.5):
        """
        Visualizes the normals of a geometry node.

        Parameters:
        - geom_node: The geometry node whose normals you want to visualize.
        - scale: How long the normal lines should be.
        """
        # Create a new NodePath to attach the lines to
        lines_np = NodePath("normals_visualization")
        
        # Create a LineSegs object to hold the lines
        lines = LineSegs()
        lines.setThickness(1.0)
        lines.setColor(1, 0, 0, 1)  # Red color for visibility

        # Iterate through the geometry to get vertex positions and normals
        geom = geom_node.node().getGeom(0)
        vdata = geom.getVertexData()
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_reader = GeomVertexReader(vdata, "normal")

        while not vertex_reader.isAtEnd():
            v = vertex_reader.getData3f()
            n = normal_reader.getData3f()
            lines.moveTo(v)
            lines.drawTo(v + n * scale)  # Draw line in the direction of the normal

        # Add the lines to the NodePath and attach it to render
        lines_np.attachNewNode(lines.create())
        lines_np.reparentTo(self.render)


    @staticmethod
    def create_mesh_data(voxel_world, voxel_size=0.5):
        vertices = []
        indices = []
        index = 0  # Keep track of the last index used

        # Define the offsets for the 8 corners of a cube
        corner_offsets = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 1]),
            np.array([1, 1, 1]),
            np.array([0, 1, 1]),
        ]
        
        # Adjust the offsets based on the voxel size
        corner_offsets = [offset * voxel_size for offset in corner_offsets]

        # Faces defined by indices into the corner_offsets, corresponding to quad vertices
        face_corners = [
            [0, 1, 5, 4],  # Front face
            [1, 2, 6, 5],  # Right face
            [2, 3, 7, 6],  # Back face
            [3, 0, 4, 7],  # Left face
            [4, 5, 6, 7],  # Top face
            [0, 3, 2, 1],  # Bottom face
        ]

        width, length, height = voxel_world.shape
        for x in range(width):
            for y in range(length):
                for z in range(height):
                    # Check if the voxel is not air
                    if voxel_world[x, y, z] != 0:
                        # Iterate over each face to determine if it should be added
                        for face in face_corners:
                            # Assume all faces are exposed for simplicity
                            # Generate vertices for this face
                            face_vertices = [corner_offsets[corner] + np.array([x, y, z]) * voxel_size for corner in face]
                            vertices.extend(face_vertices)
                            
                            # Add indices for the two triangles that make up this face
                            indices.extend([index, index+1, index+2, index, index+2, index+3])
                            index += 4  # Increment the index for the next set of vertices

        # Convert vertices to a flat list of coordinates
        vertices_flat = []
        for vertex in vertices:
            vertices_flat.extend(vertex)

        return np.array(vertices_flat, dtype=np.float32), np.array(indices, dtype=np.uint32)


    def add_mesh_to_physics(self, vertices, indices, world_x, world_y):
        terrainMesh = BulletTriangleMesh()
        for i in range(0, len(indices), 3):
            idx0, idx1, idx2 = indices[i], indices[i+1], indices[i+2]
            # Retrieve each vertex as a triplet (x, y, z)
            v0 = vertices[idx0*3:idx0*3+3]  # Fetches the x, y, z components of the vertex
            v1 = vertices[idx1*3:idx1*3+3]
            v2 = vertices[idx2*3:idx2*3+3]
            
            # Convert numpy arrays to lists if necessary
            v0 = v0.tolist() if isinstance(v0, np.ndarray) else v0
            v1 = v1.tolist() if isinstance(v1, np.ndarray) else v1
            v2 = v2.tolist() if isinstance(v2, np.ndarray) else v2

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
        scale = 0.02  # Adjust scale to control the "zoom" level of the noise
        octaves = 4  # Number of layers of noise to combine
        persistence = 7.5  # Amplitude of each octave
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
                                    base=1)  # Base can be any constant, adjust for different terrains

                # Map the noise value to a desired height range if needed
                height_map[x, y] = height

        return height_map

    def setup_crosshair(self):
        # Path to the crosshair image
        crosshair_image = 'assets/crosshair.png'
        
        # Create and position the crosshair at the center of the screen
        self.crosshair = OnscreenImage(image=crosshair_image, pos=(0, 0, 0))
        self.crosshair.setTransparency(TransparencyAttrib.MAlpha)
        self.crosshair.setScale(0.05, 1, 0.05)

    def update_physics(self, task):
        dt = globalClock.getDt()
        self.physicsWorld.doPhysics(dt)

        # Example manual collision check
        for node in self.physicsWorld.getRigidBodies():
            result = self.physicsWorld.contactTest(node)
            if result.getNumContacts() > 0:
                print(f"Collision detected for {node.getName()}")
        
        return Task.cont
    
    def update_terrain(self, task):
        self.chunk_manager.update_chunks()
        return Task.cont

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
        rotate_speed = 70  # Speed for rotating the camera, adjust as needed

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
    parser.add_argument('-g', action="store", default=-9.81, type=float)
    args = parser.parse_args()

    game = GameEngine(args)
    # Create a WindowProperties object
    props = WindowProperties()
    # Set the cursor visibility to False
    props.setCursorHidden(True)
    # Apply the properties to the main window
    game.win.requestProperties(props)
    game.run()