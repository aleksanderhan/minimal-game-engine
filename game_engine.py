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


class Block:
    def __init__(self, game_engine, position, scale, static=False):
        self.scale = scale
        self.game_engine = game_engine
        self.static = static

        # Create the voxel's collision shape
        self.voxel_shape = BulletBoxShape(Vec3(scale/2, scale/2, scale/2))

        # Create a Bullet rigid body node and attach the collision shape
        self.voxel_node = BulletRigidBodyNode('Voxel')
        self.voxel_node.addShape(self.voxel_shape)

        # Set the mass of the voxel (0 for static, >0 for dynamic)
        if static:
            self.voxel_node.setMass(0)  # Static voxel
        else:
            self.voxel_node.setMass(1.0)  # Dynamic voxel
        
        # Attach the voxel node to the scene graph
        self.voxel_np = game_engine.render.attachNewNode(self.voxel_node)
        self.voxel_np.setPythonTag("block", self)
        self.voxel_np.setPos(position)

        self.geom_np = self.create_cube_geom()

        # Load and apply the texture
        texture = self.game_engine.loader.loadTexture("assets/block.jpeg")
        self.geom_np.setTexture(texture)

        # Add the voxel to the physics world
        game_engine.physicsWorld.attachRigidBody(self.voxel_node)
    
    def make_dynamic(self):
        if self.static:
            self.voxel_node.setMass(1.0)

    def create_cube_geom(self):
        cube_np = NodePath("cube")

        # Create each face with correct orientation and position
        for i in range(6):
            face_np = self.create_face(i)
            face_np.reparentTo(cube_np)

        cube_np.reparentTo(self.voxel_np)
        cube_np.setPos(0, 0, 0)  # Ensure the cube is centered on its NodePath

        return cube_np

    def create_face(self, face_index):
        cardMaker = CardMaker(f'face{face_index}')
        cardMaker.setFrame(-self.scale/2, self.scale/2, -self.scale/2, self.scale/2)

        face_np = NodePath(cardMaker.generate())

        # Position and orient each face correctly
        if face_index == 0:  # Front
            face_np.setPos(0, -self.scale/2, 0)
            face_np.setHpr(0, 0, 0)
        elif face_index == 1:  # Back
            face_np.setPos(0, self.scale/2, 0)
            face_np.setHpr(180, 0, 0)
        elif face_index == 2:  # Right
            face_np.setPos(self.scale/2, 0, 0)
            face_np.setHpr(90, 0, 0)
        elif face_index == 3:  # Left
            face_np.setPos(-self.scale/2, 0, 0)
            face_np.setHpr(-90, 0, 0)
        elif face_index == 4:  # Top
            face_np.setPos(0, 0, self.scale/2)
            face_np.setHpr(0, -90, 0)
        elif face_index == 5:  # Bottom
            face_np.setPos(0, 0, -self.scale/2)
            face_np.setHpr(0, 90, 0)

        return face_np

class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scale = 0.5
        self.ground_height = 0

        self.chunk_size = 16
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

        print(voxel_world)
        print(voxel_world.shape)
        
        vertices, indices = self.create_mesh_data(voxel_world, 0.5)
        terrainNP = self.apply_textures_to_voxels(voxel_world, texture_positions, 2048)

        # Position the flat terrain chunk according to its world coordinates
        world_x = chunk_x * self.chunk_size
        world_y = chunk_y * self.chunk_size
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

    def apply_textures_to_voxels(self, voxel_world, texture_positions, atlas_width):
        atlas_height = 1024  # Assuming the height of the texture atlas is 1024 pixels

        # Calculate the number of different types of voxels (excluding air)
        num_voxel_types = len(texture_positions)

        # Create a GeomVertexFormat
        format = GeomVertexFormat.getV3n3t2()  # For vertices, normals, and texture coordinates
        vdata = GeomVertexData('voxel_data', format, Geom.UHStatic)
        vdata.setNumRows(4 * num_voxel_types * 6)  # Assuming each voxel can have up to 6 faces, 4 vertices per face

        # Create writers for vertices, normals, and texture coordinates
        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        normal_writer = GeomVertexWriter(vdata, 'normal')
        texcoord_writer = GeomVertexWriter(vdata, 'texcoord')

        # Iterate through the voxel_world to find voxels exposed to air
        for x in range(voxel_world.shape[0]):
            for y in range(voxel_world.shape[1]):
                for z in range(voxel_world.shape[2]):
                    voxel_type = voxel_world[x, y, z]
                    if voxel_type == 0:  # Air
                        continue

                    # Check for exposed faces and add geometry
                    for dx, dy, dz, nx, ny, nz in [(-1, 0, 0, -1, 0, 0), (1, 0, 0, 1, 0, 0), 
                                                (0, -1, 0, 0, -1, 0), (0, 1, 0, 0, 1, 0), 
                                                (0, 0, -1, 0, 0, -1), (0, 0, 1, 0, 0, 1)]:
                        if not self.is_solid_block(voxel_world, x + dx, y + dy, z + dz):
                            # Add face geometry
                            self.add_face_geometry(vertex_writer, normal_writer, texcoord_writer,
                                                x, y, z, nx, ny, nz, voxel_type, texture_positions, atlas_width, atlas_height)

        # Create a GeomTriangles object to hold the indices
        geom = Geom(vdata)
        tris = GeomTriangles(Geom.UHStatic)

        # Add indices to GeomTriangles (left as an exercise)
        # This part of the code would iterate over the vertices in groups of four (for each face),
        # and add two triangles per face to the GeomTriangles object.

        geom.addPrimitive(tris)

        # Create a GeomNode and add the Geom object
        geom_node = GeomNode('voxel_geom')
        geom_node.addGeom(geom)

        # Attach the GeomNode to a NodePath and add it to the scene
        terrainNP = NodePath(geom_node)
        terrainNP.reparentTo(self.render)
        
        texture_atlas = self.loader.loadTexture(self.texture_atlas_path)
        terrainNP.setTexture(texture_atlas)
        if self.args.debug:
            self.visualize_normals(terrainNP)
        return terrainNP
    
    def is_solid_block(self, voxel_world, x, y, z):
        """
        Checks if the voxel at the given coordinates is a solid block.

        Parameters:
        - voxel_world: A 3D numpy array representing the voxel world, where the value at each position indicates the voxel type.
        - x, y, z: The coordinates of the voxel to check.

        Returns:
        - True if the voxel is a solid block, False if it is air or the coordinates are out of bounds.
        """
        # Check if the coordinates are within the bounds of the voxel world
        if 0 <= x < voxel_world.shape[0] and 0 <= y < voxel_world.shape[1] and 0 <= z < voxel_world.shape[2]:
            return voxel_world[x, y, z] != 0  # Non-zero value indicates a solid block
        else:
            # Coordinates are out of bounds, treat as not solid (or air)
            return False

    
    def add_face_geometry(self, vertex_writer, normal_writer, texcoord_writer, x, y, z, nx, ny, nz, voxel_type, texture_positions, atlas_width, atlas_height):
        # Voxel size, assuming voxels are cubic
        voxel_size = 1

        # Calculate base vertex positions for the face
        # Offset by half the voxel size to get the corner positions
        base_vertices = [
            (x + 0.5, y + 0.5, z + 0.5),
            (x - 0.5, y + 0.5, z + 0.5),
            (x - 0.5, y - 0.5, z + 0.5),
            (x + 0.5, y - 0.5, z + 0.5),
        ]

        # Adjust vertices based on the normal to select the correct face
        vertices = []
        for vx, vy, vz in base_vertices:
            vx += nx * voxel_size / 2
            vy += ny * voxel_size / 2
            vz += nz * voxel_size / 2
            vertices.append((vx, vy, vz))

        # Write vertices and normals for the face
        for vx, vy, vz in vertices:
            vertex_writer.addData3f(vx, vy, vz)
            normal_writer.addData3f(nx, ny, nz)

        # Calculate and write texture coordinates
        # Get the UV offset for the voxel type
        u_offset, v_offset = texture_positions[voxel_type]
        texture_size = 1024  # Assuming each texture is 1024x1024 in the atlas

        # Define UV coordinates for the face, assuming the entire texture is used for the face
        uv_coords = [
            (u_offset, v_offset),
            (u_offset + texture_size / atlas_width, v_offset),
            (u_offset + texture_size / atlas_width, v_offset + texture_size / atlas_height),
            (u_offset, v_offset + texture_size / atlas_height),
        ]

        for u, v in uv_coords:
            texcoord_writer.addData2f(u, v)

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
        # Return the voxel type
        return voxel_world[tuple(voxel_indices)]
    
    @staticmethod
    def find_exposed_faces(voxel_world):
        """
        Identifies exposed faces of voxels in a voxelized world.
        
        Parameters:
        - voxel_world: A 3D numpy array where 0s represent air and non-zeros represent solid blocks.
        
        Returns:
        - exposed_faces: A boolean array of the same shape as voxel_world indicating if each voxel face is exposed.
        """
        # Expand voxel_world by 1 in each direction with zeros (air)
        padded_world = np.pad(voxel_world, 1, mode='constant', constant_values=0)
        
        # Preparing slices for comparison
        inside_world = padded_world[1:-1, 1:-1, 1:-1]
        positive_x = padded_world[2:, 1:-1, 1:-1]
        negative_x = padded_world[:-2, 1:-1, 1:-1]
        positive_y = padded_world[1:-1, 2:, 1:-1]
        negative_y = padded_world[1:-1, :-2, 1:-1]
        positive_z = padded_world[1:-1, 1:-1, 2:]
        negative_z = padded_world[1:-1, 1:-1, :-2]
        
        # Identifying exposed faces by comparing each voxel to its neighbors
        exposed_x = (inside_world > 0) & ((positive_x == 0) | (negative_x == 0))
        exposed_y = (inside_world > 0) & ((positive_y == 0) | (negative_y == 0))
        exposed_z = (inside_world > 0) & ((positive_z == 0) | (negative_z == 0))
        
        # Any voxel with at least one exposed face is considered exposed
        exposed_faces = exposed_x | exposed_y | exposed_z
        
        return exposed_faces

    @staticmethod
    def create_mesh_data(height_map, voxel_size):
        vertices = []
        indices = []
        depth, height, width = height_map.shape  # Dimensions of the height_map
        
        # Directions to check for neighboring air blocks
        neighbor_dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        def add_face(x, y, z, dir_index):
            base_index = len(vertices) // 3
            face_offsets = [
                [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],  # Positive X
                [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],  # Negative X
                [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],  # Positive Y
                [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],  # Negative Y
                [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],  # Positive Z
                [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]   # Negative Z
            ][dir_index]

            for offset in face_offsets:
                vertices.extend([(x + offset[0] - 0.5) * voxel_size, 
                                (y + offset[1] - 0.5) * voxel_size, 
                                (z + offset[2] - 0.5) * voxel_size])
            
            indices.extend([base_index, base_index + 1, base_index + 2, 
                            base_index, base_index + 2, base_index + 3])
        
        for x in range(depth):
            for y in range(height):
                for z in range(width):
                    if height_map[x, y, z] == 0:  # Skip air blocks
                        continue
                    
                    for idx, (dx, dy, dz) in enumerate(neighbor_dirs):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        # Check if neighbor is outside the bounds or is air
                        if nx < 0 or nx >= depth or ny < 0 or ny >= height or nz < 0 or nz >= width or height_map[nx, ny, nz] == 0:
                            # This voxel is exposed to air, add its face(s) to the mesh
                            add_face(x, y, z, idx)
        
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)




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