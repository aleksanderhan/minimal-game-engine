import numpy as np
import pyautogui
import argparse
import copy
import random
import time
import tqdm

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage

from panda3d.core import (
    AmbientLight, DirectionalLight, KeyboardButton,
    LineSegs, TextNode, WindowProperties,
    loadPrcFileData, GeomVertexReader, LQuaternionf
)
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletClosestHitRayResult
)
from panda3d.bullet import BulletWorld, BulletDebugNode
from panda3d.core import Vec3, Vec2
from panda3d.core import TransparencyAttrib
from panda3d.core import WindowProperties
from panda3d.core import NodePath
from panda3d.core import TransparencyAttrib
from panda3d.core import Thread

from chunk_manager import ChunkManager
from voxel import (
    DynamicArbitraryVoxelObject, create_single_voxel_mesh, create_dynamic_single_voxel_object,

)
from constants import VoxelType, voxel_type_map
from world import (
    VoxelWorld, get_center_of_hit_static_voxel, get_center_of_hit_dynamic_voxel, calculate_world_chunk_coordinates, calculate_chunk_world_position
)
from geom import create_geometry
from jit import voxel_grid_coordinates_to_index, index_to_voxel_grid_coordinates, identify_exposed_voxels, create_mesh
from util import toggle, create_voxel_type_value_color_list


random.seed(1337)

loadPrcFileData("", "load-file-type p3assimp")
loadPrcFileData("", "bullet-enable-contact-events true")
loadPrcFileData('', 'win-size 1680 1050')
loadPrcFileData("", "threading-model Cull/Draw")

def pre_warmup_jit_functions():
    print("Pre warming jit functions. Hold tight!")
    single_item_array = np.ones((1, 1, 1), np.int8)
    voxel_type_value_color_list = create_voxel_type_value_color_list()
    jit_functions = [
        lambda: index_to_voxel_grid_coordinates(0, 0, 0, 5),
        lambda: voxel_grid_coordinates_to_index(0, 0, 0, 5),
        lambda: identify_exposed_voxels(single_item_array),
        lambda: create_mesh(single_item_array, 1.0, voxel_type_value_color_list, False),
    ]

    for f in tqdm.tqdm(jit_functions):
        f()


class ObjectManager:

    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.objects = {}

    def register_object(self, 
                        object: DynamicArbitraryVoxelObject, 
                        position: Vec3, 
                        velocity = Vec3(0, 0, 0), 
                        orientation = LQuaternionf(0, 0, 0, 0),
                        ccd=False):
        self.objects[object.id] = object
        root_node_path = object.node_paths[(0, 0, 0)]

        geom_np = create_geometry(object.vertices, object.indices)
        geom_np.reparentTo(self.game_engine.render)
        geom_np.reparentTo(root_node_path)

        root_node_path.setPos(position)
        root_node_path.setQuat(orientation)

        object.set_velocity(velocity)
        if ccd:
            object.enable_ccd()

    def deregister_object(self, object):
        for _, node_path in object.node_paths.items():
            self.game_engine.physics_world.removeRigidBody(node_path.node())
            node_path.removeNode()
        del self.objects[object.id]
        del object

    def update_object(self, object, position, velocity, orientation):
        new_object = copy.deepcopy(object)
        self.deregister_object(object)
        self.register_object(new_object, position, velocity, orientation)


class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.args = args

        #self.render.setTwoSided(True)
        #self.taskMgr.popupControls()
        print("isThreadingSupported", Thread.isThreadingSupported())
        
        self.voxel_size = 0.25
        self.ground_height = self.voxel_size / 2
        self.max_height = args.n * 10
        self.chunk_size = 2 * args.n - 1

        self.chunk_manager = ChunkManager(self)
        self.object_manager = ObjectManager(self)
        self.info_display = None

        self.build_mode = False
        self.placeholder_cube: NodePath = None
        self.spawn_distance = 10

        self.selected_voxel_type_value = 0
        self.selected_voxel_type = VoxelType.AIR

        self.camera.setPos(0, 0, 50)
        self.camera.lookAt(1, 1, 1)
        self.camera_speed = 20 * self.voxel_size
        self.camera_lift_speed = 20 * self.voxel_size
        self.camera_rotate_speed = 75

        pre_warmup_jit_functions()
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

        self.accept('mouse1', self.shoot_voxel)  # Listen for left mouse click
        self.accept('f', self.create_and_place_voxel)
        self.accept('r', self.manual_raycast_test)
        self.accept('g', self.toggle_gravity)
        self.accept('b', self.toggle_build_mode)
        self.accept('i', self.print_world_info)
        self.accept('wheel_up', self.on_mouse_wheel_up)
        
        # Listen for mouse wheel down event
        self.accept('wheel_down', self.on_mouse_wheel_down)

    def setup_environment(self):
        #build_robot(self.physics_world)
        pass

    def on_mouse_wheel_up(self):
        self.selected_voxel_type_value = (self.selected_voxel_type_value + 1) % (len(VoxelType) - 1)
        self.selected_voxel_type = voxel_type_map[self.selected_voxel_type_value]
        print("selected_voxel_type_value", self.selected_voxel_type.name)

    def on_mouse_wheel_down(self):
        self.selected_voxel_type_value = (self.selected_voxel_type_value - 1) % (len(VoxelType) - 1)
        self.selected_voxel_type = voxel_type_map[self.selected_voxel_type_value]
        print("selected_voxel_type_value", self.selected_voxel_type.name)

    def print_world_info(self):
        num_surface_voxels = self.chunk_manager.get_number_of_visible_voxels()
        num_chunks_loaded = len(list(self.chunk_manager.loaded_chunks))
        print("--- World info ---")
        print("Number of loaded chunks", num_chunks_loaded)
        print("Number of surface voxels:", num_surface_voxels)

    def toggle_build_mode(self):
        self.build_mode = not self.build_mode

        if self.build_mode == True:
            raycast_result = self.cast_ray_from_camera(self.spawn_distance)

            if raycast_result.hasHit():
                voxel_center_pos = get_center_of_hit_static_voxel(raycast_result, self.voxel_size)
                hit_normal = raycast_result.getHitNormal()
                position = voxel_center_pos + hit_normal * self.voxel_size
            else:
                position = self.get_spawn_position()

            self.placeholder_cube = self._create_translucent_voxel(position)
            self.taskMgr.add(self._update_placeholder_cube, "UpdatePlaceholderCube")
        else:
            if self.placeholder_cube is not None:
                self.taskMgr.remove("UpdatePlaceholderCube")
                self.placeholder_cube.removeNode()
                self.placeholder_cube = None

    def _update_placeholder_cube(self, task: Task) -> int:
        if self.placeholder_cube is not None:
            raycast_result = self.cast_ray_from_camera(self.spawn_distance)
            
            if raycast_result.hasHit():
                hit_node = raycast_result.getNode()
                hit_normal = raycast_result.getHitNormal()

                if hit_node.static:
                    voxel_center_pos = get_center_of_hit_static_voxel(raycast_result, self.voxel_size)
                    position = voxel_center_pos + hit_normal * self.voxel_size
                    orientation = LQuaternionf.identQuat()
                    self.placeholder_cube.setPos(position)
                    self.placeholder_cube.setQuat(orientation)
                else:
                    voxel_center_pos = get_center_of_hit_dynamic_voxel(raycast_result)
                    hit_object = hit_node.getPythonTag("object")
                    position = voxel_center_pos + hit_normal * self.voxel_size
                    self.placeholder_cube.setPos(position)
                    self.placeholder_cube.setQuat(hit_object.get_orientation())
            else:
                position = self.get_spawn_position()
                orientation = self.camera.getQuat()
                self.placeholder_cube.setPos(position)
                self.placeholder_cube.setQuat(orientation)

        return Task.cont
    
    def get_spawn_position(self) -> Vec3:
        # Calculate the exact position 10 meter in front of the camera
        forward_vec = self.camera.getQuat().getForward()
        return self.camera.getPos() + forward_vec * self.spawn_distance

    def _create_translucent_voxel(self, position: Vec3) -> NodePath:
        vertices, indices, = create_single_voxel_mesh(VoxelType.STONE, self.voxel_size, self.args.debug)
        cube = create_geometry(vertices, indices)
        cube.setPos(position)
        cube.setTransparency(TransparencyAttrib.M_alpha)
        cube.reparentTo(self.render)
        return cube
        
    def create_and_place_voxel(self):
        raycast_result = self.cast_ray_from_camera()

        if raycast_result.hasHit():
            # place voxel on ground or attatch to face of other voxel
            hit_node = raycast_result.getNode()
            hit_pos = raycast_result.getHitPos()
            hit_normal = raycast_result.getHitNormal()

            if hit_node.static:
                voxel_center_pos = get_center_of_hit_static_voxel(raycast_result, self.voxel_size)
                create_position = voxel_center_pos + hit_normal * self.voxel_size
                self.create_static_voxel(create_position, self.selected_voxel_type)
            else:
                hit_object = hit_node.getPythonTag("object")
                create_position, velocity, orientation = hit_object.add_voxel(hit_pos, hit_normal, self.selected_voxel_type, self)
                create_position = get_center_of_hit_dynamic_voxel(raycast_result) + hit_normal         
                self.object_manager.update_object(hit_object, create_position, velocity, orientation)
        else:
            # place voxel in mid air
            position = self.get_spawn_position()
            orientation = self.camera.getQuat()
            velocity = Vec3(0, 0, 0)
            self.create_dynamic_voxel(position, velocity, orientation, self.selected_voxel_type)

    def create_dynamic_voxel(self, position: Vec3, velocity: Vec3, orientation: Vec3, voxel_type: VoxelType):
        object = create_dynamic_single_voxel_object(self.voxel_size, voxel_type, self.render, self.physics_world, self.args.debug)
        ccd = velocity.length() > 50
        self.object_manager.register_object(object, position, velocity, orientation, ccd)

    def create_static_voxel(self, position: Vec3, voxel_type: VoxelType):
        t0 = time.perf_counter()
        chunk_coordinates = calculate_world_chunk_coordinates(position, self.chunk_size, self.voxel_size)
        voxel_world = self.chunk_manager.get_voxel_world(chunk_coordinates)
        t1 = time.perf_counter()

        center_chunk_pos = calculate_chunk_world_position(chunk_coordinates, self.chunk_size, self.voxel_size)
        ix = int((position.x - center_chunk_pos.x) / self.voxel_size)
        iy = int((position.y - center_chunk_pos.y) / self.voxel_size)
        iz = int((position.z + self.voxel_size) / self.voxel_size) - 1
        
        t2 = time.perf_counter()

        try:
            # Set the voxel type at the calculated local coordinates
            voxel_world.set_voxel(ix, iy, iz, voxel_type)
            voxel_type_value_color_list = create_voxel_type_value_color_list()
            vertices, indices = create_mesh(voxel_world.world_array, self.voxel_size, voxel_type_value_color_list, self.args.debug)
            t3 = time.perf_counter()
            voxel_world.terrain_np = create_geometry(vertices, indices)
            t4 = time.perf_counter()
            self.chunk_manager.load_chunk(chunk_coordinates, voxel_world, vertices, indices)
        except Exception as e:
            print(e)
        finally:
            t5 = time.perf_counter()
            print("create_static_voxel:")
            print("time calculate_world_chunk_coordinates", t1-t0)
            print("time calculate_chunk_world_position", t2-t1)
            print("time create_mesh", t3-t2)
            print("time create_geometry", t4-t3)
            print("time load_chunk", t5-t4)
            print()
    
    def manual_raycast_test(self):        
        raycast_result = self.cast_ray_from_camera(10000)
        if raycast_result.hasHit():
            hit_node = raycast_result.getNode()
            hit_pos = raycast_result.getHitPos()
            hit_normal = raycast_result.getHitNormal()
            print("---------------------")
            print("hit_node", hit_node)
            
            if not hit_node.static:
                hit_object = hit_node.getPythonTag("object")
                ijk = hit_node.getPythonTag("ijk")
                node_path = hit_object.node_paths[ijk]
                position = node_path.getPos()
                print("ijk", ijk, "position", position)

            voxel_center_pos = get_center_of_hit_static_voxel(raycast_result, self.voxel_size)
            chunk_coords = calculate_world_chunk_coordinates(voxel_center_pos, self.chunk_size, self.voxel_size)
            center_chunk_pos = calculate_chunk_world_position(chunk_coords, self.chunk_size, self.voxel_size)

            info_text = f"""
                Hit position: {hit_pos}
                Hit normal: {hit_normal}
                Hit voxel center: {voxel_center_pos}
                Hit voxel static: {hit_node.static}
                Hit chunk coords: {chunk_coords}
                Hit chunk center pos: {center_chunk_pos}
            """
        else:
            info_text = "No hit detected!"
        
        if self.info_display:
            self.info_display.destroy()
    
        self.info_display = OnscreenText(text=info_text, pos=(1.4, -0.6), scale=0.05, fg=(1, 1, 1, 1), align=TextNode.ARight, mayChange=True)
        # Set up the task to remove the text
        self.doMethodLater(5, self.remove_info_text, "RemoveInfoText")
        
    def remove_info_text(self, task: Task) -> int:
        self.info_display.destroy()
        return Task.done

    def cast_ray_from_camera(self, distance: float = 10.0) -> BulletClosestHitRayResult:
        """Casts a ray from the camera to detect voxels."""
        # Get the camera's position and direction
        cam_pos = self.camera.getPos()
        cam_dir = self.camera.getQuat().getForward()
        
        # Define the ray's start and end points (within a certain distance)
        start_point = cam_pos
        end_point = cam_pos + cam_dir * distance  # Adjust the distance as needed
        
        # Perform the raycast
        return self.physics_world.rayTestClosest(start_point, end_point)
    
    def setup_lighting(self):
        self.setBackgroundColor(0.53, 0.81, 0.98, 0.6)  # Set the background to light blue
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
        self.physics_world = BulletWorld()
        self.physics_world.setGravity(next(self.acceleration_due_to_gravity))

        if self.args.debug:
            debug_node = BulletDebugNode('Debug')
            debug_np = self.render.attachNewNode(debug_node)
            debug_np.show()
            self.physics_world.setDebugNode(debug_np.node())

    def toggle_gravity(self):
        self.physics_world.setGravity(next(self.acceleration_due_to_gravity))

    def shoot_voxel(self, speed: float = 100): # TODO: choose speed by how long the user hold in the mouse button
        # Use the camera's position and orientation to shoot the bullet
        position = self.camera.getPos()
        orientation = self.camera.getQuat()
        direction = orientation.getForward()  # Get the forward direction of the camera
        velocity = direction * speed  # Adjust the speed as necessary
        # Create and shoot the bullet
        self.create_dynamic_voxel(position, velocity, orientation, self.selected_voxel_type)

    def visualize_normals(self, geom_node: NodePath, position: Vec2, scale: float = 0.5):
        """
        Visualizes the normals of a geometry node, positioning them
        correctly based on the chunk's position in the world.

        Parameters:
            - geom_node: The geometry node whose normals you want to visualize.
            - position: The chunk's position in the grid/map.
            - scale: The scale factor used for the visualization length of normals.
        """
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
            global_v = Vec3(local_v.getX() + position.x, local_v.getY() + position.y, local_v.getZ() + self.ground_height)

            # Calculate normal end point
            normal_end = global_v + n * scale * self.voxel_size

            lines.moveTo(global_v)
            lines.drawTo(normal_end)

        lines_np.attachNewNode(lines.create())
        lines_np.reparentTo(self.render)

    def create_and_apply_mesh_and_physics(self, 
                                           coordinates: tuple[int, int], 
                                           voxel_world: VoxelWorld,
                                           vertices: np.ndarray,
                                           indices: np.ndarray):
        
        terrain_np = voxel_world.terrain_np
        terrain_np.reparentTo(self.render)

        world_pos = calculate_chunk_world_position(coordinates, self.chunk_size, self.voxel_size)
        terrain_np.setPos(world_pos.x, world_pos.y, self.ground_height)

        if self.args.debug:
            terrain_np.setLightOff()
        
        if self.args.normals:
            self.visualize_normals(terrain_np, world_pos)

        # Position the flat terrain chunk according to its world coordinates
        voxel_world.terrain_node = self._create_terrain_mesh_and_physics(world_pos, vertices, indices)
    
    def _create_terrain_mesh_and_physics(self,
                                         position: Vec2,
                                         vertices: np.ndarray,
                                         indices: np.ndarray) -> BulletRigidBodyNode:
        
        terrainMesh = BulletTriangleMesh()
        
        # Loop through the indices to get triangles
        for i in range(0, len(indices), 3):
            idx0, idx1, idx2 = indices[i] * 10, indices[i+1] * 10, indices[i+2] * 10
            
            # Extract the position data from the flattened vertices array.
            v0 = vertices[idx0:idx0+3]  # Extracts x, y, z for vertex 0
            v1 = vertices[idx1:idx1+3]  # Extracts x, y, z for vertex 1
            v2 = vertices[idx2:idx2+3]  # Extracts x, y, z for vertex 2
            
            # Add the triangle to the mesh.
            terrainMesh.addTriangle(Vec3(*v0), Vec3(*v1), Vec3(*v2))

        terrain_shape = BulletTriangleMeshShape(terrainMesh, dynamic=False)
        terrain_node = BulletRigidBodyNode('Terrain')
        terrain_node.setFriction(50)
        terrain_node.addShape(terrain_shape)
        terrain_np = self.render.attachNewNode(terrain_node)

        # Set the position of the terrain's physics node to match its visual representation
        terrain_np.setPos(position.x, position.y, self.ground_height)

        self.physics_world.attachRigidBody(terrain_node)
        return terrain_node

    def setup_crosshair(self):
        # Path to the crosshair image
        crosshair_image = 'assets/crosshair.png'
        
        # Create and position the crosshair at the center of the screen
        self.crosshair = OnscreenImage(image=crosshair_image, pos=(0, 0, 0))
        self.crosshair.setTransparency(TransparencyAttrib.MAlpha)
        self.crosshair.setScale(0.05, 1, 0.05)

    def update_physics(self, task: Task) -> int:
        dt = globalClock.getDt()
        self.physics_world.doPhysics(dt)

        # Example manual collision check
        '''
        for node in self.physics_world.getRigidBodies():
            result = self.physics_world.contactTest(node)
            if result.getNumContacts() > 0:
                print(f"Collision detected for {node.getName()}")
        '''
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
    
    def mouse_task(self, task: Task) -> int:
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
                                        fg=(1, 1, 1, 1), align=TextNode.ALeft, mayChange=True)

    def update_fps_counter(self, task: Task) -> int:
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

    def get_player_chunk_coordinates(self) -> tuple[int, int]:
        player_pos = self.camera.getPos()
        return calculate_world_chunk_coordinates(player_pos, self.chunk_size, self.voxel_size)

    def move_camera_task(self, task: Task) -> int:
        dt = globalClock.getDt()
        
        # Lateral movement
        if inputState.isSet('forward'):
            self.camera.setY(self.camera, self.camera_speed * dt)
        if inputState.isSet('backward'):
            self.camera.setY(self.camera, -self.camera_speed * dt)
        if inputState.isSet('left'):
            self.camera.setX(self.camera, -self.camera_speed * dt)
        if inputState.isSet('right'):
            self.camera.setX(self.camera, self.camera_speed * dt)

        # Vertical movement
        if inputState.isSet('up'):
            self.camera.setZ(self.camera, self.camera_lift_speed * dt)
        if inputState.isSet('down'):
            self.camera.setZ(self.camera, -self.camera_lift_speed * dt)

        # Horizontal rotation
        if inputState.isSet('rotateLeft'):
            self.camera.setH(self.camera.getH() + self.camera_rotate_speed * dt)
        if inputState.isSet('rotateRight'):
            self.camera.setH(self.camera.getH() - self.camera_rotate_speed * dt)

        return Task.cont
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--terrain', action='store', default="flat")
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--normals', action="store_true", default=False)
    parser.add_argument('--profile', action="store_true", default=False)
    parser.add_argument('-g', action="store", default=-9.81, type=float)
    parser.add_argument('-n', action="store", default=16, type=int)
    parser.add_argument('-r', action="store", default=16, type=int)
    args = parser.parse_args()

    game = GameEngine(args)
    if args.debug or args.profile:
        import cProfile
        import pstats
        loadPrcFileData('', 'want-pstats 1')
        cProfile.run('game.run()', 'profile_stats')

    # Create a WindowProperties object
    props = WindowProperties()
    # Set the cursor visibility to False
    props.setCursorHidden(True)
    # Apply the properties to the main window
    game.win.requestProperties(props)
    game.run()

    if args.debug or args.profile:
        p = pstats.Stats('profile_stats')
        p.sort_stats('cumulative').print_stats(50)