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
import time
from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

from chunk_manager import ChunkManager
from helper import toggle, VoxelTools, WorldTools, DynamicArbitraryVoxelObject
from constants import color_normal_map


random.seed()

loadPrcFileData("", "load-file-type p3assimp")
loadPrcFileData("", "bullet-enable-contact-events true")
loadPrcFileData('', 'win-size 1680 1050')
loadPrcFileData("", "threading-model Cull/Draw")


class ObjectManager:

    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.objects = []

    def register_object(self, object, position):
        self.objects.append(object)

        geom_np = GameEngine.create_geometry(object.vertices, object.indices)
        geom_np.setTexture(self.game_engine.texture_atlas)
        geom_np.reparentTo(self.game_engine.render)
        geom_np.setPos(0, 0, 0)

        if self.game_engine.args.debug:
            geom_np.setLightOff()

        object_node = VoxelTools.create_dynamic_voxel_physics_node(object, self.game_engine.scale)
        object.object_node = object_node

        object_np = self.game_engine.render.attachNewNode(object_node)
        object.object_np = object_np

        object_np.setPythonTag("object", object)
        object_np.setPos(position)

        geom_np.reparentTo(object_np)

        self.game_engine.physicsWorld.attachRigidBody(object_node)


class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.texture_atlas = self.loader.loadTexture("texture_atlas.png")

        #self.render.setTwoSided(True)
        
        self.scale = 0.5
        self.ground_height = 0
        self.max_height = 50
        self.chunk_size = 8

        self.chunk_manager = ChunkManager(self)
        self.object_manager = ObjectManager(self)
        self.voxel_world_map = {}
        self.texture_paths = {
            "stone": "assets/stone.jpeg",
            "grass": "assets/grass.png"
        }

        self.camera.setPos(0, 0, 5)
        self.camera.lookAt(0, 0, 0)

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
        self.accept('r', self.manual_raycast_test)
        self.accept('g', self.toggle_gravity)

    def setup_environment(self):
        #build_robot(self.physicsWorld)
        pass
        
    def create_and_place_voxel(self):
        raycast_result = self.cast_ray_from_camera()

        if raycast_result.hasHit():
            # place voxel on ground or attatch to face of other voxel
            hit_node = raycast_result.getNode()
            hit_pos = raycast_result.getHitPos()
            hit_normal = raycast_result.getHitNormal()
            
            print("hit pos", hit_pos)
            print("hit normal", hit_normal)
            print("static", hit_node.static)
            print(hit_node)
            print()

            
            if hit_node.name == "Terrain":
                self.create_static_voxel(hit_pos, hit_normal)
            elif hit_node.name == "Voxel":

                face_center = self.get_face_center_from_hit(raycast_result)
                print("---------------")
                print("face_center", face_center)
                
                offset = self.scale / 2
                if hit_node.static:
                    self.create_static_voxel(face_center + hit_normal * offset)
                else:
                    self.create_dynamic_voxel(face_center + hit_normal * offset)

        else:
            # place voxel in mid air
            # Calculate the exact position 10 meter in front of the camera
            forward_vec = self.camera.getQuat().getForward()
            position = self.camera.getPos() + forward_vec * 5
            self.create_dynamic_voxel(position)

    def create_dynamic_voxel(self, position: Vec3, voxel_type: int=1):
        print("creating dynamic voxel, position:", position)
        object = DynamicArbitraryVoxelObject.make_single_voxel_object(self.scale, voxel_type, mass=1)
        self.object_manager.register_object(object, position)

    def create_static_voxel(self, position: Vec3, normal: Vec3, voxel_type: int=1):
        print("---------------------------------------")
        print("creating static voxel, position:", int(position.x), int(position.y), int(position.z))

        print("normal", normal)
        #  position = Vec3(position.y, -position.x, position.z)
        # Calculate chunk positions based on global position
        scaling_factor = self.chunk_size * self.scale
        print("scaling_factor", scaling_factor)

        chunk_x = int(position.x / scaling_factor)
        chunk_y = int(position.y / scaling_factor)
        print("chunk", chunk_x, chunk_y)

        # Retrieve the voxel world for the specified chunk
        voxel_world = self.voxel_world_map.get((chunk_x, chunk_y))

        # Calculate local voxel coordinates within the chunk

        voxel_x = int(position.x + normal.x)
        voxel_y = int(position.y + normal.y)
        voxel_z = int(position.z + normal.z)

        offset = int(self.chunk_size / 2 + 1)
        print("offset", offset)
        voxel_i = voxel_x if voxel_x >= 0 else voxel_x + offset
        voxel_j = voxel_y if voxel_y >= 0 else voxel_y + offset
        voxel_k = voxel_z
        
        print("voxel_x:", voxel_x, "voxel_y:", voxel_y, "voxel_z:", voxel_z)
        print("voxel_i:", voxel_i, "voxel_j:", voxel_j, "voxel_k:", voxel_k)
        print()

        

        try:
            # Set the voxel type at the calculated local coordinates
            voxel_world[voxel_x, voxel_y, voxel_z] = voxel_type
            self.chunk_manager.load_chunk(chunk_x, chunk_y)
        except Exception as e:
            print(e)

    def get_face_center_from_hit(self, raycast_result):
        hit_normal = raycast_result.getHitNormal()
        node_path = raycast_result.getNode().getPythonTag("nodePath") # TODO: FIx path
        voxel_position = node_path.getPos()  # World position of the voxel's center

        # Calculate face center based on the hit normal
        if abs(hit_normal.x) > 0.5:  # Hit on X-face
            face_center = voxel_position + Vec3(hit_normal.x * self.scale / 2, 0, 0)
        elif abs(hit_normal.y) > 0.5:  # Hit on Y-face
            face_center = voxel_position + Vec3(0, hit_normal.y * self.scale / 2, 0)
        else:  # Hit on Z-face
            face_center = voxel_position + Vec3(0, 0, hit_normal.z * self.scale / 2)

        return face_center
    
    def manual_raycast_test(self):
        raycast_result = self.cast_ray_from_camera(10000)
        if raycast_result.hasHit():
            hit_node = raycast_result.getNode()
            hit_pos = raycast_result.getHitPos()
            hit_normal = raycast_result.getHitNormal()
            print("Hit at:", hit_pos, "normal:", hit_normal, "Node:", hit_node)
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

    def apply_texture_and_physics_to_chunk(self, chunk_x, chunk_y, vertices, indices):
        world_x, world_y = self.calculate_chunk_world_position(chunk_x, chunk_y)
        terrain_np = GameEngine.create_geometry(vertices, indices)
        terrain_np.setTexture(self.texture_atlas)
        terrain_np.reparentTo(self.render)
        terrain_np.setPos(world_x, world_y, 0)

        if self.args.debug:
            terrain_np.setLightOff()
        
        if self.args.normals:
            self.visualize_normals(terrain_np, chunk_x, chunk_y)

        # Position the flat terrain chunk according to its world coordinates
        terrain_node = self.add_terrain_mesh_to_physics(vertices, indices, world_x, world_y)

        return terrain_np, terrain_node

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
            debug_node = BulletDebugNode('Debug')
            debug_np = self.render.attachNewNode(debug_node)
            debug_np.show()
            self.physicsWorld.setDebugNode(debug_np.node())

    def toggle_gravity(self):
        self.physicsWorld.setGravity(next(self.acceleration_due_to_gravity))
    
    def shoot_bullet(self, speed=100, scale=0.1, mass=0.1, color=(1, 1, 1, 1)):
        # Use the camera's position and orientation to shoot the bullet
        position = self.camera.getPos()
        direction = self.camera.getQuat().getForward()  # Get the forward direction of the camera
        velocity = direction * speed  # Adjust the speed as necessary
        
        # Create and shoot the bullet
        self.create_bullet(position, velocity, scale, mass, color)

    def shoot_big_bullet(self):
        return self.shoot_bullet(40, 1, 10, (1, 0, 0, 1))

    def create_bullet(self, position, velocity: Vec3, radius, mass, color):
        # Bullet model
        bullet_model = self.loader.loadModel("models/misc/sphere.egg")  # Use a simple sphere model
        bullet_node = BulletRigidBodyNode('Bullet')
        
        # Bullet physics
        bullet_model.setScale(radius)  # Scale down to bullet size
        bullet_shape = BulletSphereShape(radius)  # The collision shape radius
        bullet_node.setMass(mass) 
        bullet_model.setColor(*color)
        
        bullet_node.addShape(bullet_shape)
        bullet_node.setLinearVelocity(velocity)  # Set initial velocity
        
        bullet_np = self.render.attachNewNode(bullet_node)
        bullet_model.reparentTo(bullet_np)
        bullet_np.setPos(position)
        
        speed = (velocity.x**2 + velocity.y**2 + velocity.z **2)**0.5 
        if radius < 0.5 and speed > 50:
            bullet_np.node().setCcdMotionThreshold(1e-7)
            bullet_np.node().setCcdSweptSphereRadius(self.scale)
        
        self.physicsWorld.attachRigidBody(bullet_node)
        
        return bullet_np

    @staticmethod
    def create_geometry(vertices, indices, name="geom_node"):
        vdata = GeomVertexData('voxel_data', GameEngine.vertex_format_with_color(), Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        normal_writer = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        texcoord_writer = GeomVertexWriter(vdata, 'texcoord')

        for i in range(0, len(vertices), 8):  # 8 components per vertex: 3 position, 3 normal, 2 texcoord
            vertex_writer.addData3f(vertices[i], vertices[i+1], vertices[i+2])
            normal_writer.addData3f(vertices[i+3], vertices[i+4], vertices[i+5])
            texcoord_writer.addData2f(vertices[i+6], vertices[i+7])

            dx, dy, dz = vertices[i+3], vertices[i+4], vertices[i+5]
            color = color_normal_map[(dx, dy, dz)]
            color_writer.addData4f(color)

        # Create triangles using indices
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(0, len(indices), 3):
            tris.addVertices(indices[i], indices[i+1], indices[i+2])
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        geom_node = GeomNode(name)
        geom_node.addGeom(geom)

        geom_np = NodePath(geom_node)
        return geom_np
    
    @staticmethod
    def vertex_format_with_color():
        # Define a vertex array format that includes position, normal, color, and texture
        array_format = GeomVertexArrayFormat()
        array_format.addColumn("vertex", 3, Geom.NTFloat32, Geom.CPoint)
        array_format.addColumn("normal", 3, Geom.NTFloat32, Geom.CVector)
        array_format.addColumn("color", 4, Geom.NTFloat32, Geom.CColor)
        array_format.addColumn("texcoord", 2, Geom.NTFloat32, Geom.CTexcoord)

        # Create a vertex format based on the array format
        vertex_format = GeomVertexFormat()
        vertex_format.addArray(array_format)
        vertex_format = GeomVertexFormat.registerFormat(vertex_format)

        return vertex_format

    def visualize_normals(self, geom_node, chunk_x, chunk_y, scale=0.5):
        """
        Visualizes the normals of a geometry node, positioning them
        correctly based on the chunk's position in the world.

        Parameters:
        - geom_node: The geometry node whose normals you want to visualize.
        - chunk_x, chunk_y: The chunk's position in the grid/map.
        - scale: The scale factor used for the visualization length of normals.
        """
        # Assuming you have a method to calculate the chunk's world position:
        chunk_world_x, chunk_world_y = self.calculate_chunk_world_position(chunk_x, chunk_y)

        lines_np = NodePath("normals_visualization")
        lines = LineSegs()
        lines.setThickness(2.0)
        lines.setColor(1, 0, 0, 1)

        geom = geom_node.node().getGeom(0)
        vdata = geom.getVertexData()
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_reader = GeomVertexReader(vdata, "normal")

        while not vertex_reader.isAtEnd():
            local_v = vertex_reader.getData3f()
            n = normal_reader.getData3f()

            # Adjust local vertex position by chunk's world position
            global_v = Vec3(local_v.getX() + chunk_world_x, local_v.getY() + chunk_world_y, local_v.getZ())

            # Calculate normal end point
            normal_end = global_v + n * scale

            lines.moveTo(global_v)
            lines.drawTo(normal_end)

        lines_np.attachNewNode(lines.create())
        lines_np.reparentTo(self.render)

    def calculate_chunk_world_position(self, chunk_x, chunk_y):
        """
        Calculates the world position of the chunk based on its grid position.

        Parameters:
        - chunk_x, chunk_y: The chunk's position in the grid/map.
        - scale: The scale factor used in the game.

        Returns:
        Tuple[float, float]: The world coordinates of the chunk.
        """
        # Adjust these calculations based on how you define chunk positions in world space
        world_x = chunk_x * self.chunk_size * self.scale
        world_y = chunk_y * self.chunk_size * self.scale
        return world_x, world_y
    
    def add_terrain_mesh_to_physics(self, vertices, indices, world_x, world_y):
        terrainMesh = BulletTriangleMesh()
        
        # Loop through the indices to get triangles. Since vertices now include texture coords,
        # extract only the position data (first three components) for BulletPhysics.
        for i in range(0, len(indices), 3):
            idx0, idx1, idx2 = indices[i] * 8, indices[i+1] * 8, indices[i+2] * 8
            
            # Extract the position data from the flattened vertices array.
            v0 = vertices[idx0:idx0+3]  # Extracts x, y, z for vertex 0
            v1 = vertices[idx1:idx1+3]  # Extracts x, y, z for vertex 1
            v2 = vertices[idx2:idx2+3]  # Extracts x, y, z for vertex 2
            
            # Add the triangle to the mesh.
            terrainMesh.addTriangle(Vec3(*v0), Vec3(*v1), Vec3(*v2))

        terrain_shape = BulletTriangleMeshShape(terrainMesh, dynamic=False)
        terrain_node = BulletRigidBodyNode('Terrain')
        terrain_node.addShape(terrain_shape)
        terrain_np = self.render.attachNewNode(terrain_node)
        # Set the position of the terrain's physics node to match its visual representation
        terrain_np.setPos(world_x, world_y, 0)
        self.physicsWorld.attachRigidBody(terrain_node)
        return terrain_node

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
        '''
        for node in self.physicsWorld.getRigidBodies():
            result = self.physicsWorld.contactTest(node)
            if result.getNumContacts() > 0:
                print(f"Collision detected for {node.getName()}")
        '''
        return Task.cont
    
    def update_terrain(self, task):
        t0 = time.perf_counter()
        self.chunk_manager.update_chunks()
        dt = time.perf_counter() - t0
        #print(f"Updated chunks in {dt}")
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
    parser.add_argument('--normals', action="store_true", default=False)
    parser.add_argument('-g', action="store", default=-9.81, type=float)
    args = parser.parse_args()
    if args.debug:
        loadPrcFileData('', 'want-pstats 1')

    game = GameEngine(args)
    # Create a WindowProperties object
    props = WindowProperties()
    # Set the cursor visibility to False
    props.setCursorHidden(True)
    # Apply the properties to the main window
    game.win.requestProperties(props)
    game.run()