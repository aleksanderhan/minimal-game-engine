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
    BulletTriangleMesh, BulletTriangleMeshShape, BulletHeightfieldShape
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
from functools import lru_cache 
from panda3d.core import WindowProperties
from panda3d.core import CollisionNode, CollisionBox, CollisionTraverser, CollisionHandlerQueue, Point3
from panda3d.core import BitMask32
from panda3d.bullet import BulletGhostNode


random.seed()

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

    def update_chunks(self, levels=3):
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

        self.chunk_size = 24
        self.chunk_manager = ChunkManager(self)

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
        self.accept('f', self.create_and_place_voxel)

    def setup_environment(self):
        # Create the terrain mesh (both visual and physical)
        self.create_sphere((5, 5, 10))
        self.create_sphere((5, 5, 15))
        self.create_sphere((5, 5, 20))
        self.create_sphere((5, 5, 25))
        #build_robot(self.physicsWorld)

    def check_voxels_inside_volume(self, position, size=1):
        # Create a Bullet ghost node for collision detection
        ghost_node = BulletGhostNode('volume_checker')
        ghost_shape = BulletBoxShape(Vec3(size / 2, size / 2, size / 2))
        ghost_node.addShape(ghost_shape)

        # Attach the ghost node to the scene graph
        ghost_np = render.attachNewNode(ghost_node)
        ghost_np.setPos(Point3(position))

        # Add the ghost node to the physics world for collision detection
        self.physicsWorld.attachGhost(ghost_node)

        # Perform collision detection
        overlaps = len(list(filter(lambda x: x.name == "Voxel", ghost_node.getOverlappingNodes())))

        # Clean up by removing the ghost node from the physics world and scene
        self.physicsWorld.removeGhost(ghost_node)
        ghost_np.removeNode()

        # If there are overlapping nodes, then voxels are inside the volume
        return overlaps > 0

    def get_face_center_from_hit(self, raycast_result, voxel_size=1):
        hit_normal = raycast_result.getHitNormal()
        node_path = raycast_result.getNode().getPythonTag("nodePath") 
        voxel_position = node_path.getPos()  # World position of the voxel's center

        # Calculate face center based on the hit normal
        if abs(hit_normal.x) > 0.5:  # Hit on X-face
            face_center = voxel_position + Vec3(hit_normal.x * voxel_size / 2, 0, 0)
        elif abs(hit_normal.y) > 0.5:  # Hit on Y-face
            face_center = voxel_position + Vec3(0, hit_normal.y * voxel_size / 2, 0)
        else:  # Hit on Z-face
            face_center = voxel_position + Vec3(0, 0, hit_normal.z * voxel_size / 2)

        return face_center

    def create_and_place_voxel(self):
        raycast_result = self.cast_ray_from_camera()

        scale = 0.2
        if raycast_result.hasHit():
            # place voxel on ground or attatch to face of other voxel
            hit_node = raycast_result.getNode()
            hit_pos = raycast_result.getHitPos()
            hit_normal = raycast_result.getHitNormal()

            if hit_node.name == "Terrain":
                if not self.check_voxels_inside_volume(hit_pos, scale):
                    self.create_voxel(hit_pos, scale, static=True)
            elif hit_node.name == "Voxel":
                face_center = self.get_face_center_from_hit(raycast_result, scale)
                offset = scale / 2
                self.create_voxel(face_center + hit_normal * offset, scale, static=True)
        else:
            # place voxel in mid air
            # Calculate the exact position 10 meter in front of the camera
            forward_vec = self.camera.getQuat().getForward()
            position = self.camera.getPos() + forward_vec * 10
            self.create_voxel(position, scale)

    def create_voxel(self, position, scale, static=False):
        # Create the voxel's collision shape
        voxel_shape = BulletBoxShape(Vec3(scale/2, scale/2, scale/2))

        # Create a Bullet rigid body node and attach the collision shape
        voxel_node = BulletRigidBodyNode('Voxel')
        voxel_node.addShape(voxel_shape)

        # Set the mass of the voxel (0 for static, >0 for dynamic)
        if static:
            voxel_node.setMass(0)  # Static voxel
        else:
            voxel_node.setMass(1.0)  # Dynamic voxel

        # Attach the voxel node to the scene graph
        voxel_np = render.attachNewNode(voxel_node)

        voxel_np.setPythonTag("nodePath", voxel_np)
        voxel_np.setPos(position)

        # Add the voxel to the physics world
        self.physicsWorld.attachRigidBody(voxel_node)

        # Load and attach the visual model for the voxel
        voxel_model = self.loader.loadModel("models/box.egg")
        voxel_model.setScale(scale)
        voxel_model.reparentTo(voxel_np)
        voxel_model.setColor(0.5, 0.5, 0.5, 1)
        voxel_model.setPos(-scale/2, -scale/2, -scale/2)

        return voxel_node


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

    def generate_chunk(self, chunk_x, chunk_y):
        # Generate the height map for this chunk
        if self.args.terrain == "perlin":
            height_map = self.generate_perlin_height_map(chunk_x, chunk_y)
        else:
            height_map = self.generate_flat_height_map(self.chunk_size)

        # Generate mesh data from the height map
        vertices, indices = self.create_mesh_data(height_map, self.chunk_size, 7)

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

    def setup_physics(self):
        self.physicsWorld = BulletWorld()
        self.physicsWorld.setGravity(Vec3(0, 0, -9.81))

        if self.args.debug:
            debugNode = BulletDebugNode('Debug')
            debugNP = render.attachNewNode(debugNode)
            debugNP.show()
            self.physicsWorld.setDebugNode(debugNP.node())
    
    def shoot_bullet(self, speed=100, scale=0.2, mass=0.1, color=(1, 1, 1, 1)):
        # Use the camera's position and orientation to shoot the bullet
        position = self.camera.getPos()
        direction = self.camera.getQuat().getForward()  # Get the forward direction of the camera
        velocity = direction * speed  # Adjust the speed as necessary
        
        # Create and shoot the bullet
        self.create_bullet(position, velocity, scale, mass, color)

    def shoot_big_bullet(self):
        return self.shoot_bullet(30, 1, 10, (1, 0, 0, 1))

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
        bullet_model.reparentTo(bullet_np)
        
        self.physicsWorld.attachRigidBody(bullet_node)
        
        return bullet_np
    
    def create_sphere(self, position, scale=1, mass=10, color=(1, 0, 0, 1)):
        # Sphere physics
        sphereShape = BulletSphereShape(scale)  # Match this with the scale of your sphere model
        sphereNode = BulletRigidBodyNode('Sphere')
        sphereNode.addShape(sphereShape)
        sphereNode.setMass(mass)
        sphereNP = self.render.attachNewNode(sphereNode)
        sphereNP.setPos(*position)  # Adjust the height to see it fall
        self.physicsWorld.attachRigidBody(sphereNode)

        # Load the sphere model and attach it to the physics node
        sphere = self.loader.loadModel("models/misc/sphere.egg")
        sphere.reparentTo(sphereNP)  # Correctly attach the model to the NodePath
        sphere.setScale(scale)  # Adjust the scale as needed
        sphere.setColor(*color)  # Set the sphere's color

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

    @lru_cache
    def generate_perlin_height_map(self, chunk_x, chunk_y):
        scale = 0.02  # Adjust scale to control the "zoom" level of the noise
        octaves = 4  # Number of layers of noise to combine
        persistence = 1.5  # Amplitude of each octave
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
    args = parser.parse_args()

    game = GameEngine(args)
    # Create a WindowProperties object
    props = WindowProperties()
    # Set the cursor visibility to False
    props.setCursorHidden(True)
    # Apply the properties to the main window
    game.win.requestProperties(props)
    game.run()