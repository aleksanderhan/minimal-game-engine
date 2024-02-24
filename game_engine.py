from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Vec4, KeyboardButton, Vec3
from panda3d.core import WindowProperties, Geom, GeomNode, NodePath
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, TextNode
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
import numpy as np
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData
from panda3d.core import GeomVertexWriter, GeomTriangles, GeomNode, NodePath
from panda3d.core import loadPrcFileData
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Geom, GeomVertexData, GeomVertexFormat
from panda3d.core import GeomVertexWriter, GeomTriangles, GeomNode, NodePath
from panda3d.core import WindowProperties, KeyboardButton
from panda3d.core import AmbientLight, DirectionalLight, Material, LColor
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape
import tkinter as tk
from direct.actor.Actor import Actor

# Disable sync-video to unlimit the FPS
loadPrcFileData('', 'sync-video #f')
loadPrcFileData('', 'clock-mode limited')
loadPrcFileData('', 'clock-frame-rate 0')

# Use tkinter to get the screen resolution
root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.destroy()

# Set the window size to the screen resolution
#loadPrcFileData('', f'win-size {width} {height}')


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

        # Create a ground plane
        groundShape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        groundNode = BulletRigidBodyNode('Ground')
        groundNode.addShape(groundShape)
        groundNP = self.render.attachNewNode(groundNode)
        groundNP.setPos(0, 0, 0)  # Adjust if needed based on your scene setup
        self.physicsWorld.attachRigidBody(groundNode)

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
        """Create a chessboard ground with basic colors and a sphere."""
        board_size = 16  # nxn chessboard
        square_size = 2  # Each square has a size of 2x2 units
        for x in range(board_size):
            for y in range(board_size):
                cm = CardMaker(f'square_{x}_{y}')
                cm.setFrame(-square_size / 2, square_size / 2, -square_size / 2, square_size / 2)
                square = NodePath(cm.generate())
                square.reparentTo(self.render)
                square.setX((x - board_size / 2) * square_size + square_size / 2)
                square.setY((y - board_size / 2) * square_size + square_size / 2)
                square.setZ(0)  # Ensure all squares are at ground level

                # Rotate the square to lay flat
                square.setP(-90)  # Rotate 90 degrees around the X-axis

                if (x + y) % 2 == 0:
                    square.setColor(1, 1, 1, 1)  # White squares
                else:
                    square.setColor(0, 0, 0, 1)  # Black squares

        self.create_sphere((0, 0, 10))
        self.create_sphere((0, 0, 20))

        # Add some basic lighting to make sure the chessboard and sphere are well-lit
        alight = AmbientLight('alight')
        alight.setColor(LColor(0.5, 0.5, 0.5, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        dlight = DirectionalLight('dlight')
        dlight.setColor(LColor(0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlight.setDirection((-1, -1, -1))
        self.render.setLight(dlnp)

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
        """Updates the camera's orientation based on mouse movement."""
        if self.mouseWatcherNode.hasMouse():
            mouseX, mouseY = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
            if not self.firstUpdate:
                deltaX = mouseX - self.lastMouseX
                deltaY = mouseY - self.lastMouseY

                # Invert the direction of the deltaX and deltaY by multiplying them by -1
                self.cameraHeading -= deltaX * self.mouseSpeedX 
                self.cameraPitch = max(min(self.cameraPitch - deltaY * self.mouseSpeedY * -1, 90), -90)  # Invert vertical direction

                self.camera.setHpr(self.cameraHeading, self.cameraPitch, 0)
            else:
                self.firstUpdate = False

            self.lastMouseX = mouseX
            self.lastMouseY = mouseY
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
