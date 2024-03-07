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

random.seed()

loadPrcFileData("", "load-file-type p3assimp")
loadPrcFileData("", "bullet-enable-contact-events true")
loadPrcFileData('', 'win-size 1680 1050')

class ChunkManager:
    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.loaded_chunks = {}

    def get_player_chunk_pos(self):
        player_pos = self.game_engine.camera.getPos()
        chunk_x = int(player_pos.x) // self.game_engine.chunk_size
        chunk_y = int(player_pos.y) // self.game_engine.chunk_size
        return chunk_x, chunk_y

    def update_chunks(self, levels=5):
        chunk_x, chunk_y = self.get_player_chunk_pos()
        # Adjust the range to load chunks further out by one additional level
        for x in range(chunk_x - levels, chunk_x + levels):  # Increase the range by one on each side
            for y in range(chunk_y - levels, chunk_y + levels):  # Increase the range by one on each side
                if (x, y) not in self.loaded_chunks:
                    t0 = time.perf_counter()
                    self.load_chunk(x, y)
                    dt = time.perf_counter() - t0
                    print(f"Loaded chunk {x}, {y} in {dt}")

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
        self.scale = 1
        self.ground_height = 0

        self.chunk_size = 3
        self.chunk_manager = ChunkManager(self)
        self.voxel_world_map = {}
        self.current_voxel_world_chunk = None
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
        
    def get_voxel_world(self, chunk_x, chunk_y):
        if (chunk_x, chunk_y) not in self.voxel_world_map:
            max_height = 5  # Maximum height of the world
            width = self.chunk_size
            depth = self.chunk_size
            
            # Initialize an empty voxel world with air (0)
            voxel_world = np.zeros((width, depth, max_height), dtype=int)
            
            # Generate or retrieve heightmap for this chunk
            #heightmap = self.generate_flat_height_map(self.chunk_size, height=1)
            heightmap = self.generate_perlin_height_map(chunk_x, chunk_y)
            
            # Populate the voxel world based on the heightmap
            for y in range(width):
                for x in range(depth):
                    # Convert heightmap value to an integer height level
                    height = int(heightmap[x, y] * max_height)
                    
                    # Populate voxels up to this height with rock (1), ensuring not to exceed max_height
                    voxel_world[x, y, :min(height, max_height)] = 1  # Using rock (1) as an example
            
            self.voxel_world_map[(chunk_x, chunk_y)] = voxel_world
            

            return voxel_world

        return self.voxel_world_map.get((chunk_x, chunk_y))

        
    def generate_chunk(self, chunk_x, chunk_y):
        t0 = time.perf_counter()
        voxel_world = self.get_voxel_world(chunk_x, chunk_y)
        t1 = time.perf_counter()
        vertices, indices = self.create_mesh_data(voxel_world, self.scale)
        print("vertices", vertices)
        print("indices", indices)

        t2 = time.perf_counter()
        terrainNP = self.apply_textures_to_voxels(voxel_world, vertices, indices)
        t3 = time.perf_counter()
        print(f"Loaded voxel_world for chunk {chunk_x}, {chunk_y} in {t1-t0}")
        print(f"Created mesh data for chunk {chunk_x}, {chunk_y} in {t2-t1}")
        print(f"Applied textures for chunk {chunk_x}, {chunk_y} in {t3-t2}")
        if self.args.debug:
            self.visualize_normals(terrainNP, self.scale)

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

    def apply_textures_to_voxels(self, voxel_world, vertices, indices):
        texture_atlas = self.loader.loadTexture("texture_atlas.png")
        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('voxel_data', format, Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        normal_writer = GeomVertexWriter(vdata, 'normal')
        texcoord_writer = GeomVertexWriter(vdata, 'texcoord')

        # Define normals for each face of a voxel
        normals = [
            Vec3(0, -1, 0),  # Front face normal
            Vec3(1, 0, 0),   # Right face normal
            Vec3(0, 1, 0),   # Back face normal
            Vec3(-1, 0, 0),  # Left face normal
            Vec3(0, 0, 1),   # Top face normal
            Vec3(0, 0, -1)   # Bottom face normal
        ]

        # Assuming you're mapping one quad (two triangles) per face
        for i in range(0, len(indices), 6):  # 6 indices per quad (two triangles)
            face_index = i // 6 % len(normals)  # Determine which face we're working on
            normal = normals[face_index]  # Get the normal for this face

            for j in range(6):  # For each vertex in the quad
                idx = indices[i + j]
                vertex_writer.addData3f(vertices[idx * 3], vertices[idx * 3 + 1], vertices[idx * 3 + 2])
                normal_writer.addData3f(normal)  # Add the same normal for all vertices of the face

                # Assuming UV mapping is done here
                # texcoord_writer.addData2f(...)

        # Retrieve the mapping as arrays instead of a dict
        all_vertex_indices, all_voxel_types = self.generate_voxel_type_map_vectorized(voxel_world)

        # Assuming two types: type 1 uses the first half, type 2 uses the second half of the atlas
        uv_maps = {
            1: [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],  # UV coordinates for type 1
            2: [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)]   # UV coordinates for type 2
        }

        tris = GeomTriangles(Geom.UHStatic)
        vertex_count = 0
        for i in range(0, len(indices), 6):  # 6 indices per quad (two triangles)
            face_index = i // 6 % 6  # There are 6 faces per cube; adjust if your model differs
            # Get the normal for this face
            normal = normals[face_index]  # Initialize with a default normal; replace with your actual normals
            
            for j in range(6):  # For each vertex in the quad
                idx = indices[i + j]
                normal_writer.addData3f(*normal)
                
                # Determine voxel type for this vertex
                vertex_index = np.where(all_vertex_indices == vertex_count)[0]
                if vertex_index.size > 0:
                    voxel_type = all_voxel_types[vertex_index[0]]
                else:
                    voxel_type = 1  # Default to type 1 if not found
                
                # Look up UV coordinates based on voxel type
                u, v = uv_maps[voxel_type][j % 4]
                texcoord_writer.addData2f(u, v)
                vertex_count += 1

            # Define the two triangles that make up the quad
            tris.addVertices(i, i + 1, i + 2)
            tris.addVertices(i + 3, i + 4, i + 5)

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        geom_node = GeomNode('voxel_geom')
        geom_node.addGeom(geom)
        geom_np = NodePath(geom_node)
        geom_np.setTexture(texture_atlas)
        geom_np.reparentTo(self.render)

        return geom_np

    
    def generate_voxel_type_map_vectorized(self, voxel_world):
        non_empty_voxel_indices = np.argwhere(voxel_world > 0)  # Find indices of all non-empty voxels
        
        # Assuming each voxel generates a fixed number of vertices (e.g., 8 for a cube)
        # and each vertex has a unique index based on its position in the mesh,
        # we calculate base indices for the vertices of each voxel.
        # This operation replaces the get_vertex_indices_for_voxel method.
        base_indices = non_empty_voxel_indices[:, 0] * self.chunk_size * self.chunk_size * 8 + \
                    non_empty_voxel_indices[:, 1] * self.chunk_size * 8 + \
                    non_empty_voxel_indices[:, 2] * 8
        
        # Expand base indices to cover all vertices generated by each voxel.
        # For each voxel, we generate 8 vertices, so we create an array of offsets [0, 1, ..., 7]
        # and add these offsets to each base index.
        vertex_offsets = np.arange(8)
        all_vertex_indices = (base_indices[:, np.newaxis] + vertex_offsets).flatten()
        
        # Create an array of the same shape as all_vertex_indices, filled with voxel types.
        # This step assumes a direct mapping from voxel positions to types.
        voxel_types = voxel_world[non_empty_voxel_indices[:, 0], non_empty_voxel_indices[:, 1], non_empty_voxel_indices[:, 2]]
        all_voxel_types = np.repeat(voxel_types, 8)  # Repeat each type 8 times, once for each vertex generated by the voxel
        
        # Instead of a dictionary, we use two arrays: one for indices (keys) and one for types (values).
        # This format is more amenable to vectorized operations but differs from the requested dict output.
        # If a dict is absolutely required, further steps would be needed to convert these arrays into a dict,
        # which may negate some benefits of vectorization due to the overhead of dictionary creation.
        
        return all_vertex_indices, all_voxel_types

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
    def create_mesh_data(voxel_world, voxel_size):
        exposed_faces = GameEngine.identify_exposed_voxels(voxel_world)
        vertices = []
        indices = []

        voxel_offset = [
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # Front face
            [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],  # Back face
            [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],  # Left face
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],  # Right face
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],  # Top face
            [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]]   # Bottom face
        ]

        index_offset = len(vertices) // 3
        for z in range(voxel_world.shape[2]):
            for y in range(voxel_world.shape[1]):
                for x in range(voxel_world.shape[0]):
                    if not exposed_faces[x, y, z]:
                        continue

                    voxel_pos = np.array([x, y, z]) * voxel_size
                    # Generate vertices and indices for each exposed face
                    for face_vertices in voxel_offset:
                        face_indices = [index_offset + i for i in range(4)]
                        indices.extend([face_indices[0], face_indices[1], face_indices[2], face_indices[0], face_indices[2], face_indices[3]])
                        for offset in face_vertices:
                            vertex_position = voxel_pos + np.array(offset) * voxel_size
                            vertices.extend(vertex_position)
                        index_offset += 4

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    
        
    
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
            'left':  (0, -1, 0),
            'right': (0, 1, 0),
            'down':  (-1, 0, 0),
            'up':    (1, 0, 0),
            'back':  (0, 0, -1),
            'front': (0, 0, 1),
        }
        exposed_faces = np.zeros_like(voxel_world, dtype=bool)
        
        for direction, (dx, dy, dz) in shifts.items():
            shifted_world = np.roll(padded_world, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Expose face if there's air next to it (voxel value of 0 in the shifted world)
            exposed_faces |= ((shifted_world[1:-1, 1:-1, 1:-1] == 0) & (voxel_world > 0))
        
        return exposed_faces



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