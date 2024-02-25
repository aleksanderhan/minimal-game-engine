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


loadPrcFileData("", "load-file-type p3assimp")

class GameEngine(ShowBase):

    def __init__(self, args):
        super().__init__()
        self.isPerlin = args.perlin

        self.camera.setPos(0, -30, 30)
        self.camera.lookAt(0, 0, 0)

        self.setupBulletPhysics()
        self.setup_environment()
        self.setup_lighting()
        #self.build_robot()

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
        #self.taskMgr.add(self.follow_actor_task, "FollowActorTask")

        self.accept('mouse1', self.shoot_bullet)  # Listen for left mouse click

    def build_robot(self):
        # Create spherical body
        body_shape = BulletSphereShape(1)
        body_node = BulletRigidBodyNode('Body')
        body_node.addShape(body_shape)
        body_np = render.attachNewNode(body_node)
        body_np.setPos(10, 10, 2)  # Position the body
        self.physicsWorld.attachRigidBody(body_node)

        # Add visual model for the body - using a sphere from Panda3D's basic models
        sphere_model = loader.loadModel('models/misc/sphere')
        sphere_model.reparentTo(body_np)
        sphere_model.setScale(1)  # Match the BulletSphereShape's radius

        # Function to create a visual representation of a leg segment
        def create_leg_segment_visual(parent_np, scale=(0.1, 0.1, 1), pos=Vec3(0, 0, 0)):
            card_maker = CardMaker('leg_segment')
            card_maker.setFrame(-0.5, 0.5, -0.5, 0.5)  # Create a square
            leg_segment_np = parent_np.attachNewNode(card_maker.generate())
            leg_segment_np.setScale(scale)  # Scale to act as cylinder placeholder
            leg_segment_np.setPos(pos)
            return leg_segment_np

        # Create legs with physics and visual placeholders
        leg_length = 2
        for i in range(4):  # Four legs
            angle = i * (360 / 4)
            offset = Vec3(1.5 * cos(radians(angle)), 1.5 * sin(radians(angle)), 0)

            # Upper leg part
            upper_leg_shape = BulletCylinderShape(0.1, leg_length / 2, 2)
            upper_leg_node = BulletRigidBodyNode(f'UpperLeg{i}')
            upper_leg_node.addShape(upper_leg_shape)
            upper_leg_np = render.attachNewNode(upper_leg_node)
            upper_leg_np.setPos(body_np.getPos() + offset)
            self.physicsWorld.attachRigidBody(upper_leg_node)

            # Add visual placeholder for the upper leg
            create_leg_segment_visual(upper_leg_np, scale=(0.1, 0.1, leg_length / 2), pos=Vec3(0, 0, -leg_length / 4))

            # Lower leg part
            lower_leg_shape = BulletCylinderShape(0.1, leg_length / 2, 2)
            lower_leg_node = BulletRigidBodyNode(f'LowerLeg{i}')
            lower_leg_node.addShape(lower_leg_shape)
            lower_leg_np = render.attachNewNode(lower_leg_node)
            lower_leg_np.setPos(upper_leg_np.getPos() + Vec3(0, 0, -leg_length / 2))
            self.physicsWorld.attachRigidBody(lower_leg_node)

            # Add visual placeholder for the lower leg
            create_leg_segment_visual(lower_leg_np, scale=(0.1, 0.1, leg_length / 2), pos=Vec3(0, 0, -leg_length / 4))

            # Joints - no change needed here from your original code



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
        # Placeholder for terrain physics, assuming flat ground for simplicity
        groundShape = BulletRigidBodyNode('Ground')
        groundNP = self.render.attachNewNode(groundShape)
        groundNP.setPos(0, 0, -0.5)  # Adjust based on your scene
        self.physicsWorld.attachRigidBody(groundShape)

        """
        debugNode = BulletDebugNode('Debug')
        debugNP = render.attachNewNode(debugNode)
        debugNP.show()
        self.physicsWorld.setDebugNode(debugNP.node())
        """

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
    
    def create_bullet(self, position, velocity):
        # Bullet model
        bullet_model = self.loader.loadModel("models/misc/sphere.egg")  # Use a simple sphere model
        bullet_model.setScale(0.2)  # Scale down to bullet size
        
        # Bullet physics
        bullet_shape = BulletSphereShape(0.2)  # The collision shape radius
        bullet_node = BulletRigidBodyNode('Bullet')
        bullet_node.addShape(bullet_shape)
        bullet_node.setMass(0.1)
        bullet_node.setLinearVelocity(velocity)  # Set initial velocity
        
        bullet_np = self.render.attachNewNode(bullet_node)
        bullet_np.setPos(position)
        bullet_model.reparentTo(bullet_np)
        
        self.physicsWorld.attachRigidBody(bullet_node)
        
        return bullet_np

    
    def setup_environment(self):
        # Create the terrain mesh (both visual and physical)
        self.create_terrain_mesh()
        self.create_sphere((32, 32, 10))
        self.create_sphere((32, 32, 15))
        self.create_sphere((32, 32, 20))
        self.create_sphere((32, 32, 25))


    def draw_mesh_edges(self, v0, v1, v2):
        lines = LineSegs()
        lines.setColor(0, 0, 0, 1)  # Black color for the lines
        lines.setThickness(2.0)  # Thickness of the lines

        # Convert numpy.ndarray to Vec3
        v0_vec3 = Vec3(v0[0], v0[1], v0[2])
        v1_vec3 = Vec3(v1[0], v1[1], v1[2])
        v2_vec3 = Vec3(v2[0], v2[1], v2[2])

        # Define the lines for the triangle's edges
        lines.moveTo(v0_vec3)
        lines.drawTo(v1_vec3)
        lines.drawTo(v2_vec3)
        lines.drawTo(v0_vec3)

        # Create the node and attach it to the render
        line_node = lines.create()
        self.render.attachNewNode(line_node)


    def create_terrain_mesh(self):
        board_size = 64
        height_scale = 3

        # Generate the height map
        if self.isPerlin:
            height_map = self.generate_perlin_height_map(board_size)
        else:
            height_map = self.generate_flat_height_map(board_size)

        # Generate mesh data
        vertices, indices = self.create_mesh_data(height_map, board_size, height_scale)

        # Add mesh to Geom for visual representation
        self.add_mesh_to_geom(vertices, indices)

        # Add mesh to Bullet for physics simulation
        self.add_mesh_to_physics(vertices, indices)

    def create_mesh_data(self, height_map, board_size, height_scale):
        # Assuming height_map is a 2D NumPy array
        x, y = np.meshgrid(np.arange(board_size), np.arange(board_size), indexing='ij')
        z = height_map * height_scale
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

        # Efficiently compute indices for a grid mesh
        indices = np.array([[y * board_size + x,
                            (y + 1) * board_size + x,
                            y * board_size + (x + 1),
                            y * board_size + (x + 1),
                            (y + 1) * board_size + x,
                            (y + 1) * board_size + (x + 1)]
                            for x in range(board_size - 1) for y in range(board_size - 1)]).flatten()

        return vertices, indices.astype(int)


    def add_mesh_to_geom(self, vertices, indices):
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')

        # Add vertices
        for v in vertices:
            vertex.addData3f(*v)
            normal.addData3f(0, 0, 1)  # Assume up vector for simplicity
            color.addData4f(0.5, 0.5, 0.5, 1)  # Greyscale

        # Create triangles
        prim = GeomTriangles(Geom.UHStatic)

        # Iterate through indices and add vertices to the primitive
        for i in range(0, len(indices), 3):
            prim.addVertices(indices[i], indices[i+1], indices[i+2])
            # Draw edges for each triangle
            self.draw_mesh_edges(vertices[indices[i]], vertices[indices[i+1]], vertices[indices[i+2]])

        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('terrain')
        node.addGeom(geom)
        nodePath = NodePath(node)
        nodePath.reparentTo(self.render)



    def add_mesh_to_physics(self, vertices, indices):
        terrainMesh = BulletTriangleMesh()
        for i in range(0, len(indices), 3):
            v0, v1, v2 = vertices[indices[i]], vertices[indices[i+1]], vertices[indices[i+2]]
            terrainMesh.addTriangle(Vec3(*v0), Vec3(*v1), Vec3(*v2))

        terrainShape = BulletTriangleMeshShape(terrainMesh, dynamic=False)
        terrainNode = BulletRigidBodyNode('Terrain')
        terrainNode.addShape(terrainShape)
        terrainNP = self.render.attachNewNode(terrainNode)
        self.physicsWorld.attachRigidBody(terrainNode)


    def generate_perlin_height_map(self, board_size):
        """Generate a height map using Perlin noise."""
        scale = 0.1
        octaves = 3
        persistence = 1.0
        lacunarity = 0.5
        base = 0
        height_map = np.zeros((board_size, board_size))
        
        for x in range(board_size):
            for y in range(board_size):
                height_map[x][y] = noise.pnoise2(x * scale,
                                                 y * scale,
                                                 octaves=octaves,
                                                 persistence=persistence,
                                                 lacunarity=lacunarity,
                                                 repeatx=board_size,
                                                 repeaty=board_size,
                                                 base=base) * 5  # Scale height
        return height_map
    
    def generate_flat_height_map(self, board_size, height=0):
        """Generate a completely flat height map."""
        # Create a 2D NumPy array filled with the specified height value
        height_map = np.full((board_size, board_size), height)
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

    parser.add_argument('--perlin', action='store_true')
    args = parser.parse_args()

    game = GameEngine(args)
    game.run()