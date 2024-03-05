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
        flat_terrainNode, terrainNP, terrainNode = self.game_engine.generate_chunk(chunk_x, chunk_y)
        
        # Store both components in the loaded_chunks dictionary
        self.loaded_chunks[(chunk_x, chunk_y)] = (flat_terrainNode, terrainNP, terrainNode)

    def unload_chunk(self, chunk_x, chunk_y):
        chunk_data = self.loaded_chunks.pop((chunk_x, chunk_y), None)
        if chunk_data:
            flat_terrainNode, terrainNP, terrainNode = chunk_data
            # Remove the visual component from the scene
            terrainNP.removeNode()
            # Remove the physics component from the physics world
            self.game_engine.physicsWorld.removeRigidBody(terrainNode)
            self.game_engine.physicsWorld.removeRigidBody(flat_terrainNode)


class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scale = 0.5
        self.ground_height = 0

        self.chunk_size = 4
        self.chunk_manager = ChunkManager(self)
        self.heightmap_grid = {}
        self.texture_paths = {
            "stone": "assets/stone.jpeg",
            "grass": "assets/grass.png"
        }

        self.camera.setPos(0, 0, 10)
        self.camera.lookAt(5, 5, 0)

        self.create_texture_atlas()
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
        self.accept('f', self.create_and_place_block)
        self.accept('r', self.manual_raycast_test)
        self.accept('g', self.toggle_gravity)

    def setup_environment(self):
        #build_robot(self.physicsWorld)
        pass

    def create_texture_atlas(self, output_path="texture_atlas.png"):
        self.texture_atlas_path = output_path
        atlas_height = 1024  # The height for each image and the atlas
        scaled_images = []

        # Load and scale each image
        for key, path in self.texture_paths.items():
            img = Image.open(path)

            # Calculate the scaling factor to maintain aspect ratio
            aspect_ratio = img.width / img.height
            scaled_width = int(aspect_ratio * atlas_height)
            scaled_img = img.resize((scaled_width, atlas_height), Image.Resampling.LANCZOS)
            scaled_images.append(scaled_img)

        # Calculate the total width of the atlas
        total_width = sum(img.width for img in scaled_images)

        # Create a new atlas image
        atlas = Image.new('RGBA', (total_width, atlas_height), (0, 0, 0, 0))

        # Paste each scaled image into the atlas
        current_x = 0
        for img in scaled_images:
            atlas.paste(img, (current_x, 0))
            current_x += img.width

        # Save the atlas
        atlas.save(output_path)
    
    def manual_raycast_test(self):
        result = self.cast_ray_from_camera(10000)
        if result.hasHit():
            print("Hit at:", result.getHitPos())
        else:
            print("No hit detected.")

    def check_voxels_inside_volume(self, position, size=1):
        # Create a Bullet ghost node for collision detection
        ghost_node = BulletGhostNode('volume_checker')
        ghost_shape = BulletBoxShape(Vec3(size / 2, size / 2, size / 2))
        ghost_node.addShape(ghost_shape)

        # Attach the ghost node to the scene graph
        ghost_np = self.render.attachNewNode(ghost_node)
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
        node_path = raycast_result.getNode().getPythonTag("block").voxel_np
        voxel_position = node_path.getPos()  # World position of the voxel's center

        # Calculate face center based on the hit normal
        if abs(hit_normal.x) > 0.5:  # Hit on X-face
            face_center = voxel_position + Vec3(hit_normal.x * voxel_size / 2, 0, 0)
        elif abs(hit_normal.y) > 0.5:  # Hit on Y-face
            face_center = voxel_position + Vec3(0, hit_normal.y * voxel_size / 2, 0)
        else:  # Hit on Z-face
            face_center = voxel_position + Vec3(0, 0, hit_normal.z * voxel_size / 2)

        return face_center

    def create_and_place_block(self):
        raycast_result = self.cast_ray_from_camera()

        
        if raycast_result.hasHit():
            # place voxel on ground or attatch to face of other voxel
            hit_node = raycast_result.getNode()
            hit_pos = raycast_result.getHitPos()
            hit_normal = raycast_result.getHitNormal()

            if hit_node.name == "Terrain":
                if not self.check_voxels_inside_volume(hit_pos, self.scale):
                    #block = Block(self, hit_pos, self.scale, static=True)
                    pass
            elif hit_node.name == "Voxel":
                face_center = self.get_face_center_from_hit(raycast_result, self.scale)
                offset = self.scale / 2
                #block = Block(self, face_center + hit_normal * offset, self.scale, static=hit_node.static)
        else:
            # place voxel in mid air
            # Calculate the exact position 10 meter in front of the camera
            forward_vec = self.camera.getQuat().getForward()
            position = self.camera.getPos() + forward_vec * 10
            #block = Block(self, position, self.scale)

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

    def find_ground_z(self, x, y, max_search_height=1000):
        """
        Casts a ray downward at the specified x, y position to find the z position of the terrain.
        
        :param x: X coordinate
        :param y: Y coordinate
        :param max_search_height: The maximum height to search for the ground
        :return: The Z position of the ground or None if the ground is not found
        """
        start_point = Vec3(x, y, max_search_height)
        end_point = Vec3(x, y, -max_search_height)
        result = self.physicsWorld.rayTestClosest(start_point, end_point)
        if result.hasHit():
            return result.getHitPos().getZ()
        else:
            return None
        
    def get_heightmap(self, chunk_x, chunk_y):
        heightmap = self.heightmap_grid.get((chunk_x, chunk_y))
        
        if heightmap is None:
            heightmap = self.generate_perlin_height_map(chunk_x, chunk_y)
            self.heightmap_grid[(chunk_x, chunk_y)] = heightmap
            
        return heightmap
        
    def generate_chunk(self, chunk_x, chunk_y):
        # Dimensions of your voxel world
        max_height = 100

        heightmap = self.get_heightmap(chunk_x, chunk_y)
        normals = self.calculate_normals(heightmap)

        texture_positions = {
            1: (0, 0),  # Texture 1 starts at the beginning of the atlas
            2: (0.5, 0)  # Texture 2 starts halfway across the atlas
        }

        # Create a 3D grid of z-coordinates
        z_grid = np.arange(max_height).reshape(1, 1, max_height)
        # Extend the heightmap to 3D by repeating it along the z-axis
        heightmap_3d = heightmap[:, :, np.newaxis]

        # Vectorized comparison and filling to create voxel_world directly
        voxel_world = (z_grid < heightmap_3d).astype(int)
        vertices, indices = self.create_mesh_data(heightmap, self.chunk_size, 100, self.scale)
        terrainNP = self.apply_textures_to_voxels(voxel_world, texture_positions, 2048)

        # Generate the flat mesh ground
        flat_height_map = self.generate_flat_height_map(self.chunk_size, height=self.ground_height)
        flat_vertices, flat_indices = self.create_mesh_data(flat_height_map, self.chunk_size, 1, self.scale)  # Height scale is 1 for flat ground
        
        # Position the flat terrain chunk according to its world coordinates
        world_x = chunk_x * self.chunk_size
        world_y = chunk_y * self.chunk_size
        terrainNode = self.add_mesh_to_physics(vertices, indices, world_x, world_y)     
        flat_terrainNode = self.add_mesh_to_physics(flat_vertices, flat_indices, world_x, world_y)


        return flat_terrainNode, terrainNP, terrainNode

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
    
    def calculate_normals(self, height_map):
        """
        Vectorized calculation of normals for each vertex in the heightmap.

        Parameters:
        - height_map: 2D numpy array representing the heightmap.
        - scale: The scale of the heightmap in world units.

        Returns:
        - A numpy array containing normals for each vertex.
        """
        # Initialize the gradient arrays
        dhdx = np.zeros_like(height_map)
        dhdy = np.zeros_like(height_map)
        
        # Calculate gradients using differences (central for interior, forward/backward for edges)
        dhdx[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / (2 * self.scale)
        dhdy[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / (2 * self.scale)
        
        # Handle edges by extending the nearest value (simple approach)
        dhdx[0, :] = dhdx[1, :]
        dhdx[-1, :] = dhdx[-2, :]
        dhdy[:, 0] = dhdy[:, 1]
        dhdy[:, -1] = dhdy[:, -2]
        
        # Calculate normals
        normals = np.stack((-dhdx, -dhdy, np.ones_like(height_map)), axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals /= norm  # Normalize
        
        return normals

    def apply_textures_to_voxels(self, voxel_world, texture_positions, atlas_width):
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        
        prim = GeomTriangles(Geom.UHStatic)

        for x in range(voxel_world.shape[0]):
            for y in range(voxel_world.shape[1]):
                for z in range(voxel_world.shape[2]):
                    voxel_type = voxel_world[x, y, z]
                    if voxel_type != 0:  # Skip air voxels
                        cube_vertices, cube_normals, cube_indices = self.generate_cube_geometry()
                        uv_coords = self.calculate_uv_coordinates_for_cube(voxel_type, texture_positions, atlas_width)
                        
                        index_offset = vdata.getNumRows()  # Get current number of vertices
                        
                        # Add cube vertices, normals, UVs to vdata
                        for i, v in enumerate(cube_vertices):
                            vertex.addData3f(*(v + np.array([x*self.scale, y*self.scale, z*self.scale])))  # Adjust vertex position
                            normal.addData3f(*cube_normals[i % len(cube_normals)])  # Use modulo for normals
                            texcoord.addData2f(*uv_coords[i % len(uv_coords)])  # Use modulo for UVs

                        # Add indices for the cube, adjusted by the current offset
                        for i in range(0, len(cube_indices), 3):
                            prim.addVertices(cube_indices[i] + index_offset,
                                            cube_indices[i+1] + index_offset,
                                            cube_indices[i+2] + index_offset)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('terrain')
        node.addGeom(geom)
        terrainNP = NodePath(node)
        terrainNP.reparentTo(self.render)

        texture_atlas = self.loader.loadTexture(self.texture_atlas_path)
        terrainNP.setTexture(texture_atlas)
        return terrainNP
    
    def generate_cube_geometry(self):
        # Define the vertices of the cube
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],     # Front face
        ]) * self.scale * 0.5  # Scale the cube and center it

        # Define the normals for each face (assuming they are needed)
        vertices_normals = np.array([
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],  # Normals for back face
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],     # Normals for front face
            [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0],   # Normals for bottom and top faces
            [-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0],   # Normals for left and right faces
        ]) * self.scale

        # Define the indices for each face (two triangles per face)
        indices = np.array([
            0, 1, 2, 0, 2, 3,  # Back
            4, 5, 6, 4, 6, 7,  # Front
            0, 4, 5, 0, 5, 1,  # Bottom
            2, 6, 7, 2, 7, 3,  # Top
            0, 4, 7, 0, 7, 3,  # Left
            1, 5, 6, 1, 6, 2,  # Right
        ])

        return vertices, vertices_normals, indices
    
    @staticmethod
    def calculate_uv_coordinates_for_cube(voxel_type, texture_positions, atlas_width, atlas_height=1024):
        # Assume texture_positions maps voxel types to their UV offset in the atlas
        u_offset, v_offset = texture_positions[voxel_type]
        texture_size = 1024  # Assuming each texture is 1024x1024 in the atlas

        # Define UV coordinates for each cube face
        uv_coords = np.array([
            [u_offset, v_offset], [u_offset + texture_size, v_offset],
            [u_offset + texture_size, v_offset + texture_size], [u_offset, v_offset + texture_size],
        ]) / np.array([atlas_width, atlas_height])

        # Each face will use the same UV mapping, but you could adjust this if different faces use different parts of the texture
        uv_coords = np.tile(uv_coords, (6, 1))

        return uv_coords

    
    @staticmethod
    def determine_voxel_type_at_vertex(vertex, voxel_world, world_size, voxel_size):
        """
        Determines the voxel type at a given vertex position.

        Parameters:
        - vertex: The position of the vertex as a (x, y, z) tuple.
        - voxel_world: A 3D numpy array representing the voxel world, where the value at each position indicates the voxel type.
        - world_size: The size of the world in the same units as the vertex positions. This is a tuple of (width, length, height).
        - voxel_size: The size of each voxel in the same units as the vertex positions.

        Returns:
        - The voxel type at the given vertex position. Returns None if the vertex is outside the voxel world.
        """
        # Calculate the voxel indices corresponding to the vertex position
        voxel_indices = np.floor(np.array(vertex) / voxel_size).astype(int)

        # Check if the indices are within the bounds of the voxel world
        if np.any(voxel_indices < 0) or np.any(voxel_indices >= np.array(world_size) / voxel_size):
            #return None  # Vertex is outside the voxel world
            return 2
        return 1
        # Return the voxel type
        return voxel_world[tuple(voxel_indices)]

    @staticmethod
    def create_mesh_data(height_map, board_size, height_scale, voxel_size):
        """
        Generates mesh data that corresponds to a coarser granularity, matching the size of voxel faces.

        Parameters:
        - height_map: 2D array of height values.
        - board_size: The size of the board (assuming square for simplicity).
        - height_scale: Scale to apply to the height values.
        - voxel_size: The size of the side of each voxel (in mesh units), can be smaller than 1.

        Returns:
        - vertices: A flattened array of vertex coordinates.
        - indices: An array of triangle indices.
        """
        # Calculate the number of voxels along one side based on the desired board size
        # Adjust for voxel_size < 1 by scaling board_size accordingly
        num_voxels = int(np.ceil(board_size / voxel_size))

        # Generate meshgrid with adjusted size based on voxel size
        x, y = np.meshgrid(np.linspace(0, board_size, num_voxels + 1),
                        np.linspace(0, board_size, num_voxels + 1),
                        indexing='ij')

        # Adjust height_map to match the new coarser granularity
        # Resample the height_map to match the size of the voxel grid
        # This involves interpolating the height_map to fit the new grid size
        
        original_x = np.linspace(0, board_size, height_map.shape[0])
        original_y = np.linspace(0, board_size, height_map.shape[1])
        interp_func = interp2d(original_x, original_y, height_map, kind='linear')
        z = interp_func(np.linspace(0, board_size, num_voxels + 1), np.linspace(0, board_size, num_voxels + 1))

        # Apply height_scale to z values
        z *= height_scale

        # Stack x, y, and z to form vertices
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

        # Generate grid of indices for the coarser mesh
        indices_grid = np.arange((num_voxels + 1) * (num_voxels + 1)).reshape(num_voxels + 1, num_voxels + 1)

        # Quad corners indices for the coarser mesh
        top_left = indices_grid[:-1, :-1].ravel()
        top_right = indices_grid[:-1, 1:].ravel()
        bottom_left = indices_grid[1:, :-1].ravel()
        bottom_right = indices_grid[1:, 1:].ravel()

        # Forming two triangles for each quad
        triangle1 = np.stack([top_left, bottom_left, top_right], axis=-1)
        triangle2 = np.stack([top_right, bottom_left, bottom_right], axis=-1)

        # Concatenate triangles to form the square indices
        indices = np.concatenate([triangle1, triangle2], axis=1).ravel()

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
    
    def generate_flat_height_map(self, board_size, height=5):
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