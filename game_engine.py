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
import noise
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    CardMaker, Vec3, KeyboardButton, WindowProperties, NodePath,
    AmbientLight, DirectionalLight, LColor, TextNode
)
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode, BulletSphereShape,
    BulletHeightfieldShape  # This is conceptual
)
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
import numpy as np
import noise
import tkinter as tk
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape, BulletTriangleMesh, BulletTriangleMeshShape
from panda3d.core import Geom, GeomNode, GeomVertexData, GeomVertexFormat
from panda3d.core import GeomVertexWriter, GeomTriangles, NodePath
from panda3d.core import LineSegs


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

 

    def configure_window(self):
        # Configure window size and FPS settings
        loadPrcFileData('', 'sync-video #f')
        loadPrcFileData('', 'clock-mode limited')
        loadPrcFileData('', 'clock-frame-rate 0')
        root = tk.Tk()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        loadPrcFileData('', f'win-size {width} {height}')

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
        lines.setThickness(2.0)  # Adjust the thickness of the lines

        # Define the lines for the triangle's edges
        lines.moveTo(v0)
        lines.drawTo(v1)
        lines.drawTo(v2)
        lines.drawTo(v0)

        # Create the node and attach it to the render
        line_node = lines.create()
        self.render.attachNewNode(line_node)


    def create_terrain_mesh(self):
        board_size = 64  # Define the size of the terrain
        scale = 0.1  # Scale factor for Perlin noise
        height_scale = 3  # Scale factor for height to make terrain more pronounced

        # Step 1: Generate the height map
        height_map = self.generate_height_map(board_size)

        # Prepare to create visual mesh
        format = GeomVertexFormat.getV3n3c4()  # Format for vertices with position, normal, and color
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        prim = GeomTriangles(Geom.UHStatic)

        # Step 2: Create the visual mesh
        terrainMesh = BulletTriangleMesh()  # For the physical mesh

        vertex_count = 0  # Initialize a count of vertices added

        for x in range(board_size - 1):
            for y in range(board_size - 1):
                # Calculate vertex positions
                z00 = height_map[x][y] * height_scale
                z10 = height_map[x + 1][y] * height_scale
                z01 = height_map[x][y + 1] * height_scale
                z11 = height_map[x + 1][y + 1] * height_scale
                
                v0 = Vec3(x, y, z00)
                v1 = Vec3(x + 1, y, z10)
                v2 = Vec3(x, y + 1, z01)
                v3 = Vec3(x + 1, y + 1, z11)

                # After defining triangles for visual mesh, draw the edges
                self.draw_mesh_edges(v0, v1, v2)
                self.draw_mesh_edges(v2, v1, v3)

                # Add vertices for two triangles (v0, v2, v1) and (v2, v1, v3)
                for v in [v0, v1, v2, v2, v1, v3]:
                    vertex.addData3f(v.x, v.y, v.z)
                    normal.addData3f(0, 0, 1)  # Simplified normal calculation
                    color.addData4f(0.5, 0.5, 0.5, 1)  # Greyscale color
                    vertex_count += 1

                # Define triangles using the vertices just added
                base = vertex_count - 6
                prim.addVertices(base, base + 1, base + 2)
                prim.addVertices(base + 3, base + 4, base + 5)

                # Add to physical mesh
                terrainMesh.addTriangle(v0, v1, v2)
                terrainMesh.addTriangle(v2, v1, v3)

        # Create and attach the visual node
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('terrain')
        node.addGeom(geom)
        nodePath = NodePath(node)
        nodePath.reparentTo(self.render)

        # Step 3: Create the physical mesh
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