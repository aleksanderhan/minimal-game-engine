import numpy as np
import tkinter as tk
import noise
import pyautogui

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



class GameEngine(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.camera.setPos(0, -30, 30)
        self.camera.lookAt(0, 0, 0)

        self.setupBulletPhysics()
        self.setup_environment()


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

        self.accept('mouse1', self.shoot_bullet)  # Listen for left mouse click


    def setupBulletPhysics(self):
        self.physicsWorld = BulletWorld()
        self.physicsWorld.setGravity(Vec3(0, 0, -9.81))
        # Placeholder for terrain physics, assuming flat ground for simplicity
        groundShape = BulletRigidBodyNode('Ground')
        groundNP = self.render.attachNewNode(groundShape)
        groundNP.setPos(0, 0, -0.5)  # Adjust based on your scene
        self.physicsWorld.attachRigidBody(groundShape)

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
        self.create_sphere((16, 16, 10))
        self.create_sphere((16, 16, 15))
        self.create_sphere((16, 16, 20))
        self.create_sphere((16, 16, 25))
        self.create_sphere((16, 16, 30))


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
        height_map = self.generate_height_map(board_size)

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


    def generate_height_map(self, board_size):
        """Generate a height map using Perlin noise."""
        square_size = 2
        scale = 0.1
        octaves = 1
        persistence = 0.5
        lacunarity = 2.0
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


game = GameEngine()
game.run()