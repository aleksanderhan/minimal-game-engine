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
import time
from multiprocessing import Pool
from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, Geom, GeomNode, NodePath

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
        self.chunk_radius = 16
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
        chunk_data = self.pool.starmap(GameEngine.generate_chunk,
                        [(self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_world_map, x, y, self.game_engine.scale) for x, y in chunks_to_load])

        # Apply textures and physics sequentially
        t1 = time.perf_counter()
        for (x, y), (vertices, indices, voxel_world) in zip(chunks_to_load, chunk_data):
            self.game_engine.voxel_world_map[(x, y)] = voxel_world
            terrainNP, terrainNode = self.game_engine.apply_texture_and_physics(x, y, vertices, indices)
            self.loaded_chunks[(x, y)] = (terrainNP, terrainNode, len(vertices))

        # Update previously_updated_position with the current player position after loading chunks
        self.previously_updated_position = (player_chunk_x, player_chunk_y)

        t2 = time.perf_counter()
        # Unload chunks outside the new radius
        self.unload_chunks_furthest_away(player_chunk_x, player_chunk_y, self.chunk_radius)
        t3 = time.perf_counter()

        if self.game_engine.args.debug:
            print(f"Generated chunk mesh data in {t1-t0}")
            print(f"Loaded texture and physics in {t2-t1}")
            print(f"Unloaded chunks in {t3-t2}")
            print(f"Loaded vertices: {self.get_number_of_loaded_vertices()}")
            print(f"Number of visible voxels: {self.get_number_of_visible_voxels()}")

        DT = time.perf_counter() - T0
        print(f"Loaded chunks in {DT}")

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
            exposed_voxels = GameEngine.identify_exposed_voxels(world)
            result += np.count_nonzero(exposed_voxels)
        return result


    def load_chunk(self, chunk_x, chunk_y):
        # Generate the chunk and obtain both visual (terrainNP) and physics components (terrainNode)
        vertices, indices, voxel_world = GameEngine.generate_chunk(self.game_engine.chunk_size, self.game_engine.max_height, self.game_engine.voxel_world_map, chunk_x, chunk_y, self.game_engine.scale)
        terrainNP, terrainNode = self.game_engine.apply_texture_and_physics(chunk_x, chunk_y, vertices, indices)
        # Store both components in the loaded_chunks dictionary
        self.loaded_chunks[(chunk_x, chunk_y)] = (terrainNP, terrainNode)

    def unload_chunk(self, chunk_x, chunk_y):
        chunk_data = self.loaded_chunks.pop((chunk_x, chunk_y), None)
        if chunk_data:
            terrainNP, terrainNode, _ = chunk_data
            terrainNP.removeNode()
            self.game_engine.physicsWorld.removeRigidBody(terrainNode)



class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args

        #self.render.setTwoSided(True)
        
        self.scale = 1
        self.ground_height = 0
        self.max_height = 50
        self.chunk_size = 4

        self.chunk_manager = ChunkManager(self)
        self.voxel_world_map = {}
        self.texture_paths = {
            "stone": "assets/stone.jpeg",
            "grass": "assets/grass.png"
        }

        self.camera.setPos(0, 0, 3)
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
                self.create_static_voxel(hit_pos, self.scale)
            elif hit_node.name == "Voxel":
                face_center = self.get_face_center_from_hit(raycast_result, self.scale)
                offset = self.scale / 2
                if hit_node.static:
                    self.create_static_voxel(face_center + hit_normal * offset, self.scale)
                else:
                    self.create_dynamic_voxel(face_center + hit_normal * offset, self.scale)

        else:
            # place voxel in mid air
            # Calculate the exact position 10 meter in front of the camera
            forward_vec = self.camera.getQuat().getForward()
            position = self.camera.getPos() + forward_vec * 10
            self.create_static_voxel(position, self.scale)

    def create_dynamic_voxel(self, position: Vec3, voxel_type: int=1):
        # TODO implement
        pass


    def create_static_voxel(self, position: Vec3, voxel_type: int=1):
        # Convert global position to chunk coordinates
        chunk_x = int(position.x) // self.chunk_size
        chunk_y = int(position.y) // self.chunk_size

        # Convert global position to local voxel coordinates within the chunk
        voxel_x = int(position.x) % self.chunk_size
        voxel_y = int(position.y) % self.chunk_size
        voxel_z = max(int(position.z) - self.ground_height, 0)  # Ensure z is non-negative

        # Retrieve the voxel world for the specified chunk
        voxel_world = self.voxel_world_map.get((chunk_x, chunk_y))

        # Check if the z-coordinate is within bounds
        if 0 <= voxel_z < voxel_world.shape[2]:
            # Set the voxel type at the calculated local coordinates
            voxel_world[voxel_x, voxel_y, voxel_z] = voxel_type
            self.chunk_manager.load_chunk(chunk_x, chunk_y)
        else:
            print(f"Voxel z-coordinate {voxel_z} is out of bounds.")

    def get_face_center_from_hit(self, raycast_result, voxel_size=1):
        hit_normal = raycast_result.getHitNormal()
        node_path = raycast_result.getNode().getPythonTag("nodePath") # TODO: FIx path
        voxel_position = node_path.getPos()  # World position of the voxel's center

        # Calculate face center based on the hit normal
        if abs(hit_normal.x) > 0.5:  # Hit on X-face
            face_center = voxel_position + Vec3(hit_normal.x * voxel_size / 2, 0, 0)
        elif abs(hit_normal.y) > 0.5:  # Hit on Y-face
            face_center = voxel_position + Vec3(0, hit_normal.y * voxel_size / 2, 0)
        else:  # Hit on Z-face
            face_center = voxel_position + Vec3(0, 0, hit_normal.z * voxel_size / 2)

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
        
    @staticmethod
    def get_voxel_world(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y):
        if (chunk_x, chunk_y) not in voxel_world_map:
            width = chunk_size
            depth = chunk_size
            
            # Initialize an empty voxel world with air (0)
            voxel_world = np.zeros((width, depth, max_height), dtype=int)
            
            # Generate or retrieve heightmap for this chunk
            #heightmap = GameEngine.generate_flat_height_map(chunk_size, height=3)
            heightmap = GameEngine.generate_perlin_height_map(chunk_size, chunk_x, chunk_y)
            
            # Convert heightmap values to integer height levels, ensuring they do not exceed max_height
            height_levels = np.floor(heightmap).astype(int)
            height_levels = np.clip(height_levels, 1, max_height)
            adjusted_height_levels = height_levels[:-1, :-1]

            # Initialize the voxel world as zeros
            voxel_world = np.zeros((width, depth, max_height), dtype=int)

            # Create a 3D array representing each voxel's vertical index (Z-coordinate)
            z_indices = np.arange(max_height).reshape(1, 1, max_height)

            # Create a 3D boolean mask where true indicates a voxel should be set to rock (1)
            mask = z_indices < adjusted_height_levels[:,:,np.newaxis]

            # Apply the mask to the voxel world
            voxel_world[mask] = 1


            #voxel_world = np.zeros((5, 5, 5), dtype=int)
            #voxel_world[1, 0, 1] = 1    
            #voxel_world[1, 1, 1] = 1

            voxel_world_map[(chunk_x, chunk_y)] = voxel_world

            return voxel_world

        return voxel_world_map.get((chunk_x, chunk_y))

    @staticmethod
    def generate_chunk(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y, scale):
        voxel_world = GameEngine.get_voxel_world(chunk_size, max_height, voxel_world_map, chunk_x, chunk_y)
        vertices, indices = GameEngine.create_mesh_data(voxel_world, scale)
        return vertices, indices, voxel_world
        

    def apply_texture_and_physics(self, chunk_x, chunk_y, vertices, indices):
        terrainNP = self.apply_textures_to_voxels(vertices, indices)
        
        if self.args.normals:
            self.visualize_normals(terrainNP, chunk_x, chunk_y)

        # Position the flat terrain chunk according to its world coordinates
        world_x = chunk_x * self.chunk_size * self.scale
        world_y = chunk_y * self.chunk_size * self.scale
        terrainNode = self.add_mesh_to_physics(vertices, indices, world_x, world_y)
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
        #vdata = GeomVertexData('voxel_data', format, Geom.UHStatic)
        vdata = GeomVertexData('voxel_data', GameEngine.ensure_vertex_format_with_color(), Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        normal_writer = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        texcoord_writer = GeomVertexWriter(vdata, 'texcoord')

        colors = {
            (0, 1, 0): (1, 0, 0, 1), # red - front
            (0, -1, 0): (0, 1, 0, 1), # green - back
            (1, 0, 0): (0, 0, 1, 1), # blue - right
            (-1, 0, 0): (1, 1, 0, 1), # yellow -left
            (0, 0, 1): (0, 1, 1, 1), # cyan - up
            (0, 0, -1): (1, 0, 1, 1)  # magenta -down
        }


        for i in range(0, len(vertices), 8):  # 8 components per vertex: 3 position, 3 normal, 2 texcoord
            vertex_writer.addData3f(vertices[i], vertices[i+1], vertices[i+2])
            normal_writer.addData3f(vertices[i+3], vertices[i+4], vertices[i+5])
            texcoord_writer.addData2f(vertices[i+6], vertices[i+7])

            dx, dy, dz = vertices[i+3], vertices[i+4], vertices[i+5]
            color = colors[(dx, dy, dz)]
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
    def ensure_vertex_format_with_color():
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

    @staticmethod
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
        front = padded_world[x, y+1, z] == 0
        back = padded_world[x, y-1, z] == 0
        right = padded_world[x + 1, y, z] == 0
        left = padded_world[x - 1, y, z] == 0
        up = padded_world[x, y , z + 1] == 0
        down = padded_world[x, y , z - 1] == 0

        faces = ["left", "right", "down", "up", "back", "front"]
        exposed = [left, right, down, up, back, front]
        
        return [face for face, exp in zip(faces, exposed) if exp]

    @staticmethod
    def create_mesh_data(voxel_world, voxel_size):
        """Efficiently creates mesh data for exposed voxel faces.

        Args:
            voxel_world: 3D NumPy array representing voxel types.
            voxel_size: The size of each voxel in world units.

        Returns:
                vertices: A NumPy array of vertices where each group of six numbers represents the x, y, z coordinates of a vertex and its normal (nx, ny, nz).
                indices: A NumPy array of vertex indices, specifying how vertices are combined to form the triangular faces of the mesh.
        """

        exposed_voxels = GameEngine.identify_exposed_voxels(voxel_world)

        vertices = []
        indices = []
        index_counter = 0  # Track indices for each exposed face

        uv_maps = {
            1: {
                "front": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                "back": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                "right": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                "left": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                "up": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                "down": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
            }, 
            1: {
                "front": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
                "back": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
                "right": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
                "left": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
                "up": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
                "down": [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
            },
        }

        # Define offsets for each face (adjust based on your coordinate system)
        normals = {
            'front':  ( 0,  1,  0),
            'back':   (0,  -1,  0),
            'right':  ( 1,  0,  0),
            'left':   ( -1, 0,  0),
            'up':   ( 0,  0, -1),
            'down':     ( 0,  0,  1),
        }

        exposed_indices = np.argwhere(exposed_voxels)

        
        '''
        voxel_type = 1
        face_names = ["front", "right", "left", "up", "down"]

        for i in [0, 1]:
            for face_name in face_names:
                face_vertices = GameEngine.generate_face_vertices(1, 1, i, face_name, voxel_size)
                face_normals = np.tile(np.array(normals[face_name]), (4, 1))

        
                uvs = uv_maps[voxel_type][face_name]

                u, v = uvs[0]  # Cycle through the UV coordinates for each vertex

                # Append generated vertices, normals, and texture coordinates to the list
                for fv, fn in zip(face_vertices, face_normals):
                    vertices.extend([*fv, *fn, u, v])
                
                # Create indices for two triangles making up the face
                indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                    index_counter + 2, index_counter + 3, index_counter])
                
                index_counter += 4

        
        face_name = "back"
        face_vertices = GameEngine.generate_face_vertices(1, 1, 1, face_name, voxel_size)
        face_normals = np.tile(np.array(normals[face_name]), (4, 1))

        uvs = uv_maps[voxel_type][face_name]

        u, v = uvs[0]  # Cycle through the UV coordinates for each vertex

        # Append generated vertices, normals, and texture coordinates to the list
        for fv, fn in zip(face_vertices, face_normals):
            vertices.extend([*fv, *fn, u, v])
        
        # Create indices for two triangles making up the face
        indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
            index_counter + 2, index_counter + 3, index_counter])
        
        index_counter += 4
        
        
        '''
        
        for x, y, z in exposed_indices:
            exposed_faces = GameEngine.check_surrounding_air_vectorized(voxel_world, x, y, z)
            j = 0
            for face_name, normal in normals.items():
                if face_name in exposed_faces:
                    # Generate vertices for this face
                    face_vertices = GameEngine.generate_face_vertices(x, y, z, face_name, voxel_size)
                    face_normals = np.tile(np.array(normal), (4, 1))

                    voxel_type = voxel_world[x, y, z]
                    uvs = uv_maps[voxel_type][face_name]

                    u, v = uvs[j % 4]  # Cycle through the UV coordinates for each vertex

                    # Append generated vertices, normals, and texture coordinates to the list
                    for fv, fn in zip(face_vertices, face_normals):
                        vertices.extend([*fv, *fn, u, v])
                    
                    # Create indices for two triangles making up the face
                    indices.extend([index_counter, index_counter + 1, index_counter + 2,  # First triangle
                        index_counter + 2, index_counter + 3, index_counter])
                    
                    index_counter += 4
                    j += 1
        
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)
    
    @staticmethod
    def generate_face_vertices(x, y, z, face_name, voxel_size):
        """
        Generates vertices and normals for a given voxel face.

        Args:
            x, y, z: Coordinates of the voxel in the voxel grid.
            dx, dy, dz: Direction vector of the face, points along the normal
            voxel_size: Size of the voxel.

        Returns:
            face_vertices: A list of vertex positions for the face.
        """


        offset_list = {
            "front": [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)],
            "back": [(1, -1, -1), (1, -1, 1), (-1, -1, 1), (-1, -1, -1)],
            "right": [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
            "left": [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)],
            "up": list(reversed([(-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1)])),
            "down": list(reversed([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)])),
        }[face_name]
        


        voxel_half = voxel_size / 2

        # Calculate vertex positions
        face_vertices = [
            [(x + offset[0]*voxel_half)*voxel_size, (y + offset[1]*voxel_half)*voxel_size, (z + offset[2]*voxel_half)*voxel_size]
            for offset in offset_list
        ]

        return np.array(face_vertices)
    

    @staticmethod
    def noop_transform(face):
        return face
    
    @staticmethod
    def rotate_face_90_degrees_ccw_around_z(face):
        # Rotate each point in the face 90 degrees counter-clockwise around the Z axis
        return [(y, -x, z) for x, z, y in face]
    
    @staticmethod
    def rotate_face_90_degrees_ccw_around_x(face):
        # Rotate each point in the face 90 degrees counter-clockwise around the X axis
        return [(x, -z, y) for x, y, z in face]

    @staticmethod
    def rotate_face_90_degrees_ccw_around_y(face):
        # Rotate each point in the face 90 degrees counter-clockwise around the Y axis
        return [(z, y, -x) for x, y, z in face]

    
    @staticmethod
    def identify_exposed_voxels(voxel_world):
        """
        Identifies a voxel exposed to air and returns a same shaped boolean np array with the result.
        True means it is exposed to air, False means it's not.

        Parameters:
            - voxel_world: a 3D numpy array representing the voxel types as integers in the world
        """
        # Pad the voxel world with zeros (air) on all sides
        padded_world = np.pad(voxel_world, pad_width=1, mode='constant', constant_values=0)
        
        # Create shifted versions of the world for all six directions
        shifts = {
            'front': (0, 1, 0),
            'back':  (0, -1, 0),
            'right': (1, 0, 0),
            'left':  (-1, 0, 0),
            'up':    (0, 0, 1),
            'down':  (0, 0, -1),
        }
        exposed_faces = np.zeros_like(voxel_world, dtype=bool)
        
        for direction, (dx, dy, dz) in shifts.items():
            shifted_world = np.roll(padded_world, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Expose face if there's air next to it (voxel value of 0 in the shifted world)
            exposed_faces |= ((shifted_world[1:-1, 1:-1, 1:-1] == 0) & (voxel_world > 0))
        
        return exposed_faces

    def add_mesh_to_physics(self, vertices, indices, world_x, world_y):
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
    
    @staticmethod
    def generate_perlin_height_map(chunk_size, chunk_x, chunk_y):
        scale = 0.05  # Adjust scale to control the "zoom" level of the noise
        octaves = 6  # Number of layers of noise to combine
        persistence = 0.5  # Amplitude of each octave
        lacunarity = 2.0  # Frequency of each octave

        height_map = np.zeros((chunk_size + 1, chunk_size + 1))

        # Calculate global offsets
        global_offset_x = chunk_x * chunk_size
        global_offset_y = chunk_y * chunk_size

        for x in range(chunk_size + 1):
            for y in range(chunk_size + 1):
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
                height_map[x, y] = height * 30

        return height_map
    
    @staticmethod
    def generate_flat_height_map(board_size, height=1):
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