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
from multiprocessing import Pool
from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

from helper import toggle, VoxelTools, WorldTools
from constants import color_normal_map


random.seed()

loadPrcFileData("", "load-file-type p3assimp")
loadPrcFileData("", "bullet-enable-contact-events true")
loadPrcFileData('', 'win-size 1680 1050')
loadPrcFileData("", "threading-model Cull/Draw")

class ChunkManager:
    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.loaded_chunks = {}
        self.pool = Pool(processes=6)
        self.previously_updated_position = None  # Initialize with None or with the player's starting position
        self.inner_radius = 8
        self.chunk_radius = 12
        self.num_chunks = 4*int(3.14*self.chunk_radius**2)

    def get_player_chunk_pos(self):
        player_pos = self.game_engine.camera.getPos()
        chunk_x = int(player_pos.x / self.game_engine.scale) // self.game_engine.chunk_size
        chunk_y = int(player_pos.y / self.game_engine.scale) // self.game_engine.chunk_size
        return chunk_x, chunk_y

    def update_chunks(self):
        T0 = time.perf_counter()
        player_chunk_x, player_chunk_y = self.get_player_chunk_pos()


        if self.previously_updated_position: 
            distance_from_center = ((player_chunk_x - self.previously_updated_position[0])**2 + 
                                    (player_chunk_y - self.previously_updated_position[1])**2)**0.5
            if distance_from_center <= self.inner_radius:
                return  # Player still within inner radius, no loading needed
            
        chunks_to_load = self.identify_chunks_to_load(player_chunk_x, player_chunk_y, self.chunk_radius)
        
        # Use multiprocessing to generate chunks
        t0 = time.perf_counter()
        chunk_data = self.pool.starmap(WorldTools.generate_chunk,
                        [(self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_world_map, x, y, self.game_engine.scale) for x, y in chunks_to_load])

        # Apply textures and physics sequentially
        t1 = time.perf_counter()

        create_world_DT = 0
        crete_mesh_DT = 0
        for (x, y), (vertices, indices, voxel_world, create_world_dt, crete_mesh_dt) in zip(chunks_to_load, chunk_data):
            create_world_DT += create_world_dt
            crete_mesh_DT += crete_mesh_dt
            self.game_engine.voxel_world_map[(x, y)] = voxel_world
            terrainNP, terrainNode = self.game_engine.apply_texture_and_physics_to_chunk(x, y, vertices, indices)
            self.loaded_chunks[(x, y)] = (terrainNP, terrainNode, len(vertices))

        # Update previously_updated_position with the current player position after loading chunks
        self.previously_updated_position = (player_chunk_x, player_chunk_y)

        t2 = time.perf_counter()
        # Unload chunks outside the new radius
        self.unload_chunks_furthest_away(player_chunk_x, player_chunk_y, self.chunk_radius)
        t3 = time.perf_counter()

        if self.game_engine.args.debug:
            print(f"Generated chunk mesh data in {t1-t0}")
            print(f"    Created world in {create_world_DT}")
            print(f"    Created mesh in {crete_mesh_DT}")
            print(f"Loaded texture and physics in {t2-t1}")
            print(f"Unloaded chunks in {t3-t2}")
            print(f"Loaded vertices: {self.get_number_of_loaded_vertices()}")
            print(f"Number of visible voxels: {self.get_number_of_visible_voxels()}")

        DT = time.perf_counter() - T0
        print(f"Loaded chunks in {DT}")
        print()

    def identify_chunks_to_load(self, player_chunk_x, player_chunk_y, chunk_radius):
        # Initialize an empty list to store the coordinates of chunks that need to be loaded.
        chunks_to_load = []

        # Iterate through all possible chunk coordinates around the player within the chunk_radius.
        for x in range(player_chunk_x - chunk_radius, player_chunk_x + chunk_radius + 1):
            for y in range(player_chunk_y - chunk_radius, player_chunk_y + chunk_radius + 1):
                
                # Calculate the distance from the current chunk to the player's chunk position.
                distance_from_player = ((x - player_chunk_x)**2 + (y - player_chunk_y)**2)**0.5
                
                # Check if the chunk is within the specified radius and not already loaded.
                if distance_from_player <= chunk_radius and (x, y) not in self.loaded_chunks:
                    # If the chunk meets the criteria, add it to the list of chunks to load.
                    if (x, y) not in self.loaded_chunks:
                        chunks_to_load.append((x, y))

        # Return the list of chunks that need to be loaded.
        return chunks_to_load

    def unload_chunks_furthest_away(self, player_chunk_x, player_chunk_y, chunk_radius):
        # Calculate distance for each loaded chunk and keep track of their positions and distances
        chunk_distances = [
            (chunk_pos, ((chunk_pos[0] - player_chunk_x)**2 + (chunk_pos[1] - player_chunk_y)**2))
            for chunk_pos in self.loaded_chunks.keys()
        ]
        
        # Sort chunks by their distance in descending order (furthest first)
        chunk_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Select the furthest self.num_chunks to unload
        chunks_to_unload = list(filter(lambda x: x[1] > chunk_radius, chunk_distances[:len(self.loaded_chunks) - self.num_chunks]))
        
        # Unload these chunks
        for chunk_pos, _ in chunks_to_unload:
            self.unload_chunk(*chunk_pos)
            
        print(f"Unloaded {len(chunks_to_unload)} furthest chunks.")
        
    def get_number_of_loaded_vertices(self):
        result = 0
        for _, _, num_vertices in self.loaded_chunks.values():
            result += num_vertices
        return result
    
    def get_number_of_visible_voxels(self):
        result = 0
        for key in self.loaded_chunks.keys():
            world = self.game_engine.voxel_world_map.get(key)
            exposed_voxels = VoxelTools.identify_exposed_voxels(world)
            result += np.count_nonzero(exposed_voxels)
        return result


    def load_chunk(self, chunk_x, chunk_y):
        # Generate the chunk and obtain both visual (terrainNP) and physics components (terrainNode)
        vertices, indices, _, _, _ = WorldTools.generate_chunk(self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_world_map, chunk_x, chunk_y, self.game_engine.scale)
        terrainNP, terrainNode = self.game_engine.apply_texture_and_physics_to_chunk(chunk_x, chunk_y, vertices, indices)
        # Store both components in the loaded_chunks dictionary
        self.loaded_chunks[(chunk_x, chunk_y)] = (terrainNP, terrainNode, len(vertices)) # TODO: Use Chunk dataclass

    def unload_chunk(self, chunk_x, chunk_y):
        chunk_data = self.loaded_chunks.pop((chunk_x, chunk_y), None)
        if chunk_data:
            terrainNP, terrainNode, _ = chunk_data
            terrainNP.removeNode()
            self.game_engine.physicsWorld.removeRigidBody(terrainNode)



class DynamicArbitraryVoxelObject:

    def __init__(self):

        self.object_array = None


    @staticmethod
    def make_single_voxel_object(position: Vec3, scale, voxel_type=1):
        pass




class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args

        #self.render.setTwoSided(True)
        
        self.scale = 0.5
        self.ground_height = 0
        self.max_height = 50
        self.chunk_size = 8

        self.chunk_manager = ChunkManager(self)
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

            if hit_node.name == "Terrain":
                self.create_static_voxel(hit_pos)
            elif hit_node.name == "Voxel":
                face_center = self.get_face_center_from_hit(raycast_result)
                offset = self.scale / 2
                if hit_node.static:
                    self.create_static_voxel(face_center + hit_normal * offset)
                else:
                    self.create_dynamic_voxel(face_center + hit_normal * offset)

        else:
            # place voxel in mid air
            # Calculate the exact position 10 meter in front of the camera
            forward_vec = self.camera.getQuat().getForward()
            position = self.camera.getPos() + forward_vec * 10
            self.create_static_voxel(position)

    def create_dynamic_voxel(self, position: Vec3, voxel_type: int=1):
        # TODO implement
        pass


    def create_static_voxel(self, position: Vec3, voxel_type: int=1):
        print("creating static voxel, position:", position)
        #  position = Vec3(position.y, -position.x, position.z)
        # Calculate chunk positions based on global position
        chunk_x = int(position.x / (self.chunk_size * self.scale))
        chunk_y = int(position.y / (self.chunk_size * self.scale))

        # Calculate local voxel coordinates within the chunk
        local_x = position.x % (self.chunk_size * self.scale)
        local_y = position.y % (self.chunk_size * self.scale)
        voxel_x = int(local_x / self.scale)
        voxel_y = int(local_y / self.scale)
        voxel_z = int(position.z / self.scale)

        print("x:", voxel_x, "y:", voxel_y, "z:", voxel_z)


        # Retrieve the voxel world for the specified chunk
        voxel_world = self.voxel_world_map.get((chunk_x, chunk_y))

        # Check if the z-coordinate is within bounds
        if 0 <= voxel_z < voxel_world.shape[2]:
            # Set the voxel type at the calculated local coordinates
            voxel_world[voxel_x, voxel_y, voxel_z] = voxel_type
            self.chunk_manager.load_chunk(chunk_x, chunk_y)
        else:
            print(f"Voxel z-coordinate {voxel_z} is out of bounds.")

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
        terrainNP = self.apply_textures_to_voxels(vertices, indices)
        
        if self.args.normals:
            self.visualize_normals(terrainNP, chunk_x, chunk_y)

        # Position the flat terrain chunk according to its world coordinates
        world_x = chunk_x * self.chunk_size * self.scale
        world_y = chunk_y * self.chunk_size * self.scale
        terrainNode = self.add_terrain_mesh_to_physics(vertices, indices, world_x, world_y)
        terrainNP.setPos(world_x, world_y, 0)

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

    def apply_textures_to_voxels(self, vertices, indices):
        texture_atlas = self.loader.loadTexture("texture_atlas.png")
        format = GeomVertexFormat.getV3n3t2()  # Ensure format includes texture coordinates
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

        geom_node = GeomNode('voxel_geom')
        geom_node.addGeom(geom)
        geom_np = NodePath(geom_node)
        geom_np.setTexture(texture_atlas)
        geom_np.reparentTo(self.render)

        if self.args.debug:
            geom_np.setLightOff()

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
        chunk_world_x, chunk_world_y = self.calculate_chunk_world_position(chunk_x, chunk_y, scale)

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

    def calculate_chunk_world_position(self, chunk_x, chunk_y, scale):
        """
        Calculates the world position of the chunk based on its grid position.

        Parameters:
        - chunk_x, chunk_y: The chunk's position in the grid/map.
        - scale: The scale factor used in the game.

        Returns:
        Tuple[float, float]: The world coordinates of the chunk.
        """
        # Adjust these calculations based on how you define chunk positions in world space
        world_x = chunk_x * self.chunk_size * scale
        world_y = chunk_y * self.chunk_size * scale
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

        terrainShape = BulletTriangleMeshShape(terrainMesh, dynamic=False)
        terrainNode = BulletRigidBodyNode('Terrain')
        terrainNode.addShape(terrainShape)
        terrainNP = self.render.attachNewNode(terrainNode)
        # Set the position of the terrain's physics node to match its visual representation
        terrainNP.setPos(world_x, world_y, 0)
        self.physicsWorld.attachRigidBody(terrainNode)
        return terrainNode

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